"""
engine/backtest.py - Walk-Forward 歷史回測、DSR 抗過擬合檢定與 5 分位數單調性檢驗
===================================================================================
Provides Walk-Forward out-of-sample backtesting, Deflated Sharpe Ratio (DSR) anti-overfitting test,
multi-horizon directional accuracy (5D~252D), 5-quantile monotonicity analysis (Q1-Q5),
confusion matrix diagnostics, and MAE/MFE trade excursions.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import kurtosis, norm, pearsonr, skew, spearmanr

try:
    from hk_quant_system.config.settings import (
        DEFAULT_ALMGREN_GAMMA,
        DEFAULT_BROKER_COMM_RATE,
        DEFAULT_MIN_BROKER_COMM_HKD,
        MIN_REQUIRED_DAILY_BARS,
        calculate_hkex_microstructure_cost,
        get_candidate_tickers,
        get_hk_tick_size,
        normalize_hk_ticker,
        round_to_hk_tick,
    )
    from hk_quant_system.data.loader import (
        compute_active_daily_returns,
        fetch_fundamental_data,
        fetch_hk_macro_regime,
        safe_download_ohlcv,
    )
    from hk_quant_system.engine.alpha import compute_factor_exposure_and_ic_weights, compute_wilder_rsi
    from hk_quant_system.engine.risk import (
        compute_deflated_sharpe_ratio,
        compute_ledoit_wolf_cov,
        optimize_portfolio_weights,
    )
except (ImportError, ValueError):
    from config.settings import (
        DEFAULT_ALMGREN_GAMMA,
        DEFAULT_BROKER_COMM_RATE,
        DEFAULT_MIN_BROKER_COMM_HKD,
        MIN_REQUIRED_DAILY_BARS,
        calculate_hkex_microstructure_cost,
        get_candidate_tickers,
        get_hk_tick_size,
        normalize_hk_ticker,
        round_to_hk_tick,
    )
    from data.loader import (
        compute_active_daily_returns,
        fetch_fundamental_data,
        fetch_hk_macro_regime,
        safe_download_ohlcv,
    )
    from engine.alpha import compute_factor_exposure_and_ic_weights, compute_wilder_rsi
    from engine.risk import (
        compute_deflated_sharpe_ratio,
        compute_ledoit_wolf_cov,
        optimize_portfolio_weights,
    )

def run_strategy_backtest(
    price_df: pd.DataFrame,
    lookback: int = 60,
    rebalance_freq: int = 20,
    max_single_weight: float = 0.35,
    cost_rate: float = 0.0016,
    rf_rate: float = 0.04,
    alloc_method: str = "hrp",
    num_trials: int = 5,
    cov_estimator: str = "ledoit_wolf",
    max_turnover: float = 0.40,
    suspension_df: pd.DataFrame | None = None,
    adv_map: dict[str, float] | None = None,
    total_capital: float = 300_000.0,
    broker_comm_rate: float = DEFAULT_BROKER_COMM_RATE,
    min_broker_comm_hkd: float = DEFAULT_MIN_BROKER_COMM_HKD,
    almgren_gamma: float = DEFAULT_ALMGREN_GAMMA,
    min_required_bars: int = MIN_REQUIRED_DAILY_BARS,
) -> dict[str, Any] | None:
    """
    Institutional Out-of-Sample Walk-Forward Portfolio Backtest with strict t-1 history windows.
    """
    try:
        if price_df is None or price_df.empty:
            return None

        valid_cols = []
        excluded_cols = []
        for col in price_df.columns:
            col_s = price_df[col].dropna()
            if suspension_df is not None and col in suspension_df.columns:
                active_cnt = int((~suspension_df[col].reindex(col_s.index, fill_value=False)).sum())
            else:
                active_cnt = int((col_s.pct_change().abs() > 1e-9).sum())
            if len(col_s) >= min_required_bars and active_cnt >= max(60, int(min_required_bars * 0.7)):
                valid_cols.append(col)
            else:
                excluded_cols.append(col)

        if not valid_cols:
            return None

        clean_prices = price_df[valid_cols].sort_index().ffill(limit=3).dropna(how="any")
        if len(clean_prices) < max(min_required_bars, lookback + 20):
            return None

        raw_rets = clean_prices.pct_change().iloc[1:]
        masked_rets, _ = compute_active_daily_returns(
            clean_prices,
            suspension_df=suspension_df[valid_cols] if (suspension_df is not None and set(valid_cols).issubset(suspension_df.columns)) else None
        )

        if len(raw_rets) <= lookback + 10:
            return None

        num_assets = len(valid_cols)
        equal_weights = np.full(num_assets, 1.0 / num_assets)

        oos_rets = raw_rets.iloc[lookback:].fillna(0.0)
        strat_daily_rets = []
        bench_daily_rets = []
        turnover_records = []

        curr_strat_w = equal_weights.copy()
        curr_bench_w = equal_weights.copy()

        for i in range(len(oos_rets)):
            t_idx = lookback + i
            day_ret_vec = oos_rets.iloc[i].values

            if i % rebalance_freq == 0:
                hist_slice = masked_rets.iloc[t_idx - lookback : t_idx]
                prev_w_input = None if i == 0 else curr_strat_w
                opt_dict = optimize_portfolio_weights(
                    hist_slice,
                    max_single_weight=max_single_weight,
                    method=alloc_method,
                    cov_estimator=cov_estimator,
                    prev_weights=prev_w_input,
                    max_turnover=max_turnover if i > 0 else None,
                )
                target_w = np.array([opt_dict[col] for col in valid_cols], dtype=float)
                turnover = float(np.sum(np.abs(target_w - curr_strat_w)))
                turnover_records.append(turnover)

                if turnover > 1e-5:
                    total_rebal_cost_hkd = 0.0
                    for idx_c, col in enumerate(valid_cols):
                        w_diff = abs(float(target_w[idx_c] - curr_strat_w[idx_c]))
                        trade_hkd = w_diff * float(total_capital)
                        if trade_hkd > 100.0:
                            adv_c = float(adv_map.get(col, 30_000_000.0)) if adv_map else 30_000_000.0
                            sig_c = float(hist_slice[col].std()) if not np.isnan(hist_slice[col].std()) else 0.02
                            c_res = calculate_hkex_microstructure_cost(
                                trade_value_hkd=trade_hkd,
                                adv_20d_hkd=adv_c,
                                daily_volatility=sig_c,
                                broker_comm_rate=broker_comm_rate,
                                min_broker_comm_hkd=min_broker_comm_hkd,
                                gamma=almgren_gamma,
                            )
                            total_rebal_cost_hkd += c_res["total_frictions_hkd"]
                    tx_cost = total_rebal_cost_hkd / max(float(total_capital), 1.0)
                else:
                    tx_cost = 0.0

                curr_strat_w = target_w
            else:
                tx_cost = 0.0

            port_r = float(np.dot(day_ret_vec, curr_strat_w)) - tx_cost
            bench_r = float(np.dot(day_ret_vec, curr_bench_w))
            strat_daily_rets.append(port_r)
            bench_daily_rets.append(bench_r)

            drifted_strat = curr_strat_w * (1.0 + day_ret_vec)
            curr_strat_w = drifted_strat / (np.sum(drifted_strat) + 1e-12)

            drifted_bench = curr_bench_w * (1.0 + day_ret_vec)
            curr_bench_w = drifted_bench / (np.sum(drifted_bench) + 1e-12)

        strat_series = pd.Series(strat_daily_rets, index=oos_rets.index)
        bench_series = pd.Series(bench_daily_rets, index=oos_rets.index)

        strategy_nav = (1.0 + strat_series).cumprod()
        benchmark_nav = (1.0 + bench_series).cumprod()

        n_days = max(1, len(strat_series))
        total_return_strat = float(strategy_nav.iloc[-1] - 1.0)
        cagr_strat = float(strategy_nav.iloc[-1] ** (252.0 / n_days) - 1.0)
        ann_vol_strat = float(strat_series.std() * np.sqrt(252))
        sharpe_strat = float((cagr_strat - rf_rate) / (ann_vol_strat + 1e-9))

        rolling_max = strategy_nav.cummax()
        drawdowns = (strategy_nav - rolling_max) / rolling_max
        max_dd = float(drawdowns.min())

        bench_rolling_max = benchmark_nav.cummax()
        bench_drawdowns = (benchmark_nav - bench_rolling_max) / bench_rolling_max

        dsr_stats = compute_deflated_sharpe_ratio(strat_series, rf_rate=rf_rate, num_trials=num_trials)

        return {
            "dates": strategy_nav.index,
            "strategy_nav": strategy_nav,
            "benchmark_nav": benchmark_nav,
            "drawdown_series": drawdowns,
            "bench_drawdown_series": bench_drawdowns,
            "total_return": total_return_strat,
            "cagr": cagr_strat,
            "ann_volatility": ann_vol_strat,
            "sharpe": sharpe_strat,
            "max_drawdown": max_dd,
            "lookback_days": lookback,
            "rebalance_freq": rebalance_freq,
            "alloc_method": alloc_method,
            "cov_estimator": cov_estimator,
            "avg_rebalance_turnover": float(np.mean(turnover_records)) if turnover_records else 0.0,
            "excluded_tickers": excluded_cols,
            "dsr_prob": dsr_stats["dsr_prob"],
            "dsr_pvalue": dsr_stats["p_value"],
            "dsr_significant": dsr_stats["is_significant"],
        }
    except Exception:
        return None


def evaluate_historical_prediction_accuracy(
    ticker: str,
    forward_horizon: int = 21,
    history_period: str = "3y",
    atr_stop_mult: float = 2.0,
    signal_z_threshold: float = 0.25,
    trade_capital_hkd: float = 100_000.0,
    broker_comm_rate: float = DEFAULT_BROKER_COMM_RATE,
    min_broker_comm_hkd: float = DEFAULT_MIN_BROKER_COMM_HKD,
    almgren_gamma: float = DEFAULT_ALMGREN_GAMMA,
) -> tuple[dict[str, Any] | None, str | None]:
    """
    Vectorized Multi-Horizon Historical Signal Validation & Accuracy Engine.
    """
    try:
        k = max(2, int(forward_horizon))
        period_map = {"6mo": "2y", "1y": "2y", "3y": "5y", "5y": "5y"}
        dl_period = period_map.get(history_period, "5y")

        df_full, matched_ticker = safe_download_ohlcv(
            ticker, period=dl_period, interval="1d", min_rows=MIN_REQUIRED_DAILY_BARS
        )
        if df_full is None or matched_ticker is None:
            return None, f"無法獲取 {ticker} 足夠的歷史日線數據（需至少 {MIN_REQUIRED_DAILY_BARS} 個有效交易日）"

        susp_mask = df_full['Is_Suspended'] if 'Is_Suspended' in df_full.columns else (df_full['Volume'] <= 0)
        df = df_full.loc[~susp_mask].copy()
        if len(df) < MIN_REQUIRED_DAILY_BARS:
            return None, f"{matched_ticker} 剔除停牌日後有效交易日僅 {len(df)} 根，不足 {MIN_REQUIRED_DAILY_BARS} 根最低門檻。"

        hsi_df, _ = safe_download_ohlcv("^HSI", period=dl_period, interval="1d", min_rows=60)

        fundamental = fetch_fundamental_data(matched_ticker)
        macro = fetch_hk_macro_regime()

        target_bars_map = {"6mo": 126, "1y": 252, "3y": 756, "5y": 1260}
        req_bars = target_bars_map.get(history_period, 756)

        factor_res = compute_factor_exposure_and_ic_weights(
            df,
            fundamental_z=fundamental["val_qual_zscore"],
            macro_z=macro["macro_zscore"],
            horizon=k,
            ic_window=max(252, req_bars),
        )

        z_signal_lagged = factor_res["composite_z_series"].shift(1)
        prob_up_lagged = factor_res["prob_up_series"].shift(1)
        score_lagged = (prob_up_lagged - 0.5) * 200.0

        close_arr = df['Close'].to_numpy(copy=True)
        open_arr = df['Open'].to_numpy(copy=True) if 'Open' in df.columns else close_arr
        high_arr = df['High'].to_numpy(copy=True)
        low_arr = df['Low'].to_numpy(copy=True)
        raw_close_arr = df['Raw_Close'].to_numpy(copy=True) if 'Raw_Close' in df.columns else close_arr
        vol_arr = df['Volume'].to_numpy(copy=True)

        tr_s = pd.concat([
            df['High'] - df['Low'],
            (df['High'] - df['Close'].shift(1)).abs(),
            (df['Low'] - df['Close'].shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr_pct_lagged = (tr_s.ewm(alpha=1.0 / 14.0, min_periods=10, adjust=False).mean() / (df['Close'] + 1e-9)).shift(1).bfill().values

        adv_20 = float(pd.Series(raw_close_arr * vol_arr).rolling(20, min_periods=5).mean().dropna().iloc[-1])
        sigma_20 = float(pd.Series(close_arr).pct_change().rolling(20, min_periods=5).std().dropna().iloc[-1])
        micro_rt = calculate_hkex_microstructure_cost(
            trade_value_hkd=trade_capital_hkd,
            adv_20d_hkd=adv_20,
            daily_volatility=sigma_20,
            broker_comm_rate=broker_comm_rate,
            min_broker_comm_hkd=min_broker_comm_hkd,
            gamma=almgren_gamma,
        )
        round_trip_cost_frac = (2.0 * micro_rt["total_cost_bps"]) / 10000.0

        target_bars_map = {"6mo": 126, "1y": 252, "3y": 756, "5y": 1260}
        req_bars = target_bars_map.get(history_period, 756)

        n_total = len(df)
        max_valid_end = n_total - k
        if max_valid_end <= 40:
            k = max(5, min(k, n_total // 3))
            max_valid_end = n_total - k

        start_idx = max(30, max_valid_end - req_bars)
        eval_indices = np.arange(start_idx, max_valid_end)
        n_eval = len(eval_indices)
        if n_eval < 15:
            return None, f"評估窗口內有效前瞻樣本數 ({n_eval}) 不足，請縮短前瞻週期或延長歷史回測區間。"

        offset_grid = eval_indices[:, None] + np.arange(k)[None, :]
        entry_opens = open_arr[eval_indices]
        window_highs = high_arr[offset_grid]
        window_lows = low_arr[offset_grid]
        window_closes = close_arr[offset_grid]

        long_excursions_high = (window_highs / (entry_opens[:, None] + 1e-9)) - 1.0
        long_excursions_low = (window_lows / (entry_opens[:, None] + 1e-9)) - 1.0
        raw_fwd_returns = (window_closes[:, -1] / (entry_opens + 1e-9)) - 1.0

        z_eval = z_signal_lagged.values[eval_indices]
        prob_eval = prob_up_lagged.values[eval_indices]
        score_eval = score_lagged.values[eval_indices]
        atr_pct_eval = atr_pct_lagged[eval_indices]
        eval_dates = df.index[eval_indices]

        stop_dist_vec = np.clip(float(atr_stop_mult) * atr_pct_eval, 0.015, 0.35)

        sig_sign = np.where(z_eval >= 0.0, 1.0, -1.0)
        mae_vec = np.where(
            sig_sign > 0,
            np.minimum(0.0, np.min(long_excursions_low, axis=1)),
            np.minimum(0.0, -np.max(long_excursions_high, axis=1)),
        )
        mfe_vec = np.where(
            sig_sign > 0,
            np.maximum(0.0, np.max(long_excursions_high, axis=1)),
            np.maximum(0.0, -np.min(long_excursions_low, axis=1)),
        )

        directional_low_path = np.where(
            sig_sign[:, None] > 0,
            long_excursions_low,
            -long_excursions_high,
        )
        hit_stop_matrix = directional_low_path <= (-stop_dist_vec[:, None])
        stopped_out_mask = np.any(hit_stop_matrix, axis=1)
        first_stop_day = np.argmax(hit_stop_matrix, axis=1) + 1
        actual_holding_days = np.where(stopped_out_mask, first_stop_day, k).astype(float)

        unstopped_dir_ret = sig_sign * raw_fwd_returns
        realized_dir_ret = np.where(stopped_out_mask, -stop_dist_vec, unstopped_dir_ret) - round_trip_cost_frac

        active_sig_mask = np.abs(z_eval) >= float(signal_z_threshold)
        if np.sum(active_sig_mask) < 8:
            active_sig_mask = np.ones_like(z_eval, dtype=bool)

        correct_dir = (np.sign(z_eval[active_sig_mask]) == np.sign(raw_fwd_returns[active_sig_mask]))
        hit_rate = float(np.mean(correct_dir)) if len(correct_dir) > 0 else 0.5

        roll_ic_window = max(60, min(120, n_eval // 3))
        s_z = pd.Series(z_eval, index=eval_dates)
        s_ret = pd.Series(raw_fwd_returns, index=eval_dates)

        rank_z = s_z.rank(pct=True)
        rank_ret = s_ret.rank(pct=True)
        # Expanding-mean centering eliminates Stambaugh/Hurwicz local-window downward bias on multi-day returns
        u_c = rank_z - rank_z.expanding(min_periods=15).mean()
        v_c = rank_ret - rank_ret.expanding(min_periods=15).mean()
        cov_uv = (u_c * v_c).rolling(roll_ic_window, min_periods=20).mean()
        std_u = np.sqrt((u_c * u_c).rolling(roll_ic_window, min_periods=20).mean())
        std_v = np.sqrt((v_c * v_c).rolling(roll_ic_window, min_periods=20).mean())
        rolling_rank_ic = (cov_uv / (std_u * std_v + 1e-9)).clip(-1.0, 1.0).dropna()

        overall_pearson_ic, _ = pearsonr(z_eval, raw_fwd_returns) if np.std(z_eval) > 1e-9 else (0.0, 1.0)
        overall_rank_ic, _ = spearmanr(z_eval, raw_fwd_returns) if np.std(z_eval) > 1e-9 else (0.0, 1.0)

        mean_ic = float(rolling_rank_ic.mean()) if not rolling_rank_ic.empty else float(overall_rank_ic)
        std_ic = float(rolling_rank_ic.std()) if len(rolling_rank_ic) > 2 else 0.15
        ic_ir = float(mean_ic / (std_ic + 1e-9))

        try:
            q_labels = pd.qcut(s_z.rank(method="first"), q=5, labels=["Q1 (極空)", "Q2 (偏空)", "Q3 (中性)", "Q4 (偏多)", "Q5 (極多)"])
        except Exception:
            q_labels = pd.Series(["Q3 (中性)"] * n_eval, index=eval_dates)

        quantile_names = ["Q1 (極空)", "Q2 (偏空)", "Q3 (中性)", "Q4 (偏多)", "Q5 (極多)"]
        quantile_summary = []
        quantile_cum_curves: dict[str, pd.Series] = {}

        daily_equiv_ret = s_ret / float(k)
        for q_name in quantile_names:
            q_mask = (q_labels == q_name)
            q_rets = s_ret[q_mask]
            mean_r = float(q_rets.mean()) if not q_rets.empty else 0.0
            win_r = float((q_rets > 0).mean()) if not q_rets.empty else 0.0
            ann_r = float((1.0 + mean_r) ** (252.0 / k) - 1.0) if mean_r > -0.99 else -0.99

            q_daily_stream = daily_equiv_ret.where(q_mask, 0.0)
            quantile_cum_curves[q_name] = (1.0 + q_daily_stream).cumprod()

            quantile_summary.append({
                "quantile": q_name,
                "count": int(q_mask.sum()),
                "mean_forward_return_pct": mean_r * 100.0,
                "annualized_return_pct": ann_r * 100.0,
                "positive_rate_pct": win_r * 100.0,
            })

        q_means = [row["mean_forward_return_pct"] for row in quantile_summary]
        monotonicity_corr, _ = spearmanr([1, 2, 3, 4, 5], q_means)
        q5_q1_spread_pct = q_means[-1] - q_means[0]

        active_rets = realized_dir_ret[active_sig_mask]
        wins = active_rets[active_rets > 0]
        losses = active_rets[active_rets < 0]

        gross_profit = float(np.sum(wins)) if len(wins) > 0 else 0.0
        gross_loss = float(np.abs(np.sum(losses))) if len(losses) > 0 else 1e-6
        profit_factor = float(gross_profit / gross_loss) if gross_loss > 1e-5 else (3.0 if gross_profit > 0 else 1.0)

        avg_win = float(np.mean(wins)) if len(wins) > 0 else 0.0
        avg_loss = float(np.abs(np.mean(losses))) if len(losses) > 0 else 1e-6
        win_loss_ratio = float(avg_win / avg_loss) if avg_loss > 1e-5 else 1.0
        expected_val_pct = float(np.mean(active_rets) * 100.0) if len(active_rets) > 0 else 0.0
        avg_hold_days = float(np.mean(actual_holding_days[active_sig_mask]))

        strat_daily_equiv = pd.Series(
            np.where(active_sig_mask, realized_dir_ret / float(k), 0.0),
            index=eval_dates
        )
        strat_cum_nav = (1.0 + strat_daily_equiv).cumprod()
        strat_ann_ret = float(strat_cum_nav.iloc[-1] ** (252.0 / max(1, n_eval)) - 1.0)

        if hsi_df is not None and not hsi_df.empty:
            hsi_aligned = hsi_df['Close'].reindex(eval_dates).ffill().bfill()
            hsi_total = float(hsi_aligned.iloc[-1] / (hsi_aligned.iloc[0] + 1e-9) - 1.0)
            hsi_ann_ret = float((1.0 + hsi_total) ** (252.0 / max(1, n_eval)) - 1.0) if hsi_total > -0.99 else 0.0
        else:
            hsi_ann_ret = 0.02
        annualized_alpha = strat_ann_ret - hsi_ann_ret

        pred_bull = z_eval[active_sig_mask] >= 0.0
        actual_bull = (raw_fwd_returns[active_sig_mask] > 0.0) & (~(pred_bull & stopped_out_mask[active_sig_mask]))

        tp = int(np.sum(pred_bull & actual_bull))
        fp = int(np.sum(pred_bull & (~actual_bull)))
        tn = int(np.sum((~pred_bull) & (~actual_bull)))
        fn = int(np.sum((~pred_bull) & actual_bull))

        fpr = float(fp / max(1, fp + tn))
        precision = float(tp / max(1, tp + fp))
        recall = float(tp / max(1, tp + fn))
        stop_trigger_rate = float(np.mean(stopped_out_mask[active_sig_mask]))

        x_score = score_eval
        y_ret_pct = raw_fwd_returns * 100.0
        if np.std(x_score) > 1e-6:
            slope, intercept = np.polyfit(x_score, y_ret_pct, 1)
            r_sq = float(overall_pearson_ic ** 2)
        else:
            slope, intercept, r_sq = 0.0, 0.0, 0.0

        scatter_df = pd.DataFrame({
            "date": eval_dates.strftime("%Y-%m-%d"),
            "pred_zscore": np.round(z_eval, 3),
            "pred_prob_pct": np.round(prob_eval * 100.0, 1),
            "pred_score": np.round(x_score, 1),
            "forward_return_pct": np.round(y_ret_pct, 2),
            "mae_pct": np.round(mae_vec * 100.0, 2),
            "mfe_pct": np.round(mfe_vec * 100.0, 2),
            "stopped_out": stopped_out_mask,
            "quantile": q_labels.astype(str).values,
        })

        return {
            "ticker": matched_ticker,
            "forward_horizon": k,
            "history_period": history_period,
            "sample_size": int(np.sum(active_sig_mask)),
            "total_bars_evaluated": n_eval,
            "hit_rate": hit_rate,
            "mean_ic": mean_ic,
            "overall_rank_ic": float(overall_rank_ic),
            "overall_pearson_ic": float(overall_pearson_ic),
            "ic_ir": ic_ir,
            "annualized_alpha": annualized_alpha,
            "strategy_ann_return": strat_ann_ret,
            "benchmark_ann_return": hsi_ann_ret,
            "profit_factor": min(profit_factor, 9.99),
            "win_loss_ratio": min(win_loss_ratio, 9.99),
            "expected_return_pct": expected_val_pct,
            "avg_holding_days": avg_hold_days,
            "monotonicity_score": float(0.0 if np.isnan(monotonicity_corr) else monotonicity_corr),
            "q5_q1_spread_pct": q5_q1_spread_pct,
            "quantile_summary": quantile_summary,
            "quantile_cum_curves": quantile_cum_curves,
            "rolling_rank_ic": rolling_rank_ic,
            "confusion_matrix": {
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "fpr": fpr,
                "precision": precision,
                "recall": recall,
                "stop_trigger_rate": stop_trigger_rate,
            },
            "regression": {
                "slope": float(slope),
                "intercept": float(intercept),
                "r_squared": r_sq,
            },
            "scatter_df": scatter_df,
            "round_trip_cost_bps": micro_rt["total_cost_bps"] * 2.0,
        }, None
    except Exception as e:
        return None, str(e)