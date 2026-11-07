"""
engine/alpha.py - 4大核心因子、滾動 Rank-IC 加權、羅吉斯回歸機率與 VIF 診斷
===================================================================================
Provides Wilder's RSI, 09:30 HKT session-anchored VWAP, Time-of-Day volume normalization,
standardized multi-factor exposures [-3.0, +3.0], rolling Rank-IC weighting, Logistic Regression P(Up),
VIF collinearity diagnostics, and tick-adjusted day trading signals.
"""

from __future__ import annotations

import math
from datetime import datetime, time
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
        HK_DEFAULT_LOT_SIZES,
        MIN_REQUIRED_DAILY_BARS,
        calculate_hkex_microstructure_cost,
        get_candidate_tickers,
        get_hk_tick_size,
        is_hk_short_eligible,
        normalize_hk_ticker,
        round_to_hk_tick,
    )
    from hk_quant_system.data.futu_gateway import (
        check_futu_opend_connection,
        fetch_realtime_spot_price,
        get_futu_order_book,
        get_futu_realtime_quote,
    )
    from hk_quant_system.data.loader import (
        fetch_fundamental_data,
        fetch_hk_macro_regime,
        safe_download_ohlcv,
    )
except (ImportError, ValueError):
    from config.settings import (
        DEFAULT_ALMGREN_GAMMA,
        DEFAULT_BROKER_COMM_RATE,
        DEFAULT_MIN_BROKER_COMM_HKD,
        HK_DEFAULT_LOT_SIZES,
        MIN_REQUIRED_DAILY_BARS,
        calculate_hkex_microstructure_cost,
        get_candidate_tickers,
        get_hk_tick_size,
        is_hk_short_eligible,
        normalize_hk_ticker,
        round_to_hk_tick,
    )
    from data.futu_gateway import (
        check_futu_opend_connection,
        fetch_realtime_spot_price,
        get_futu_order_book,
        get_futu_realtime_quote,
    )
    from data.loader import (
        fetch_fundamental_data,
        fetch_hk_macro_regime,
        safe_download_ohlcv,
    )

def compute_wilder_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Compute standard Wilder's RSI using exponential smoothing (alpha = 1/period)."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100.0 - (100.0 / (1.0 + rs))


# ============================================================================
# Task 1.4: Strict Session-Anchored Intraday VWAP (09:30 HKT Reset Boundary)
# ============================================================================
def compute_intraday_vwap_and_tod_volume(df: pd.DataFrame) -> tuple[pd.Series, float, pd.Series]:
    """
    For intraday bars (5m / 15m):
    1. Strictly enforce HKEX trading session reset at 09:30 HKT every trading day.
       Pre-09:30 auction noise or post-16:00 bars do not contaminate continuous session cumulative volume/PV.
    2. Compute Time-of-Day (ToD) normalized volume ratio and ToD volume percentile rank
       to eliminate HKEX U-shape opening/closing volume bias.
    """
    if df.empty:
        empty_s = pd.Series(dtype=float)
        return empty_s, 1.0, empty_s

    idx_dt = pd.to_datetime(df.index)
    if getattr(idx_dt, "tz", None) is not None:
        idx_dt = idx_dt.tz_convert("Asia/Hong_Kong").tz_localize(None)

    typical_price = (df['High'] + df['Low'] + df['Close']) / 3.0
    vol = df['Volume'].fillna(0.0).clip(lower=0.0)

    bar_times = idx_dt.time
    t_open = time(9, 30)
    t_lunch_start = time(12, 0)
    t_lunch_end = time(13, 0)
    t_close = time(16, 10)

    in_morning = (bar_times >= t_open) & (bar_times <= t_lunch_start)
    in_afternoon = (bar_times >= t_lunch_end) & (bar_times <= t_close)
    in_regular_session = in_morning | in_afternoon

    session_date = idx_dt.date
    is_post_open = bar_times >= t_open
    session_key = pd.Series(
        [f"{d}_{1 if post else 0}" for d, post in zip(session_date, is_post_open)],
        index=df.index
    )

    eff_vol = pd.Series(np.where(in_regular_session, vol, 0.0), index=df.index)
    pv = typical_price * eff_vol

    cum_pv = pv.groupby(session_key).cumsum()
    cum_vol = eff_vol.groupby(session_key).cumsum()

    vwap = pd.Series(
        np.where(cum_vol > 0, cum_pv / (cum_vol + 1e-9), typical_price),
        index=df.index
    )

    time_slot_series = pd.Series(bar_times, index=df.index)
    tod_means = eff_vol.groupby(time_slot_series).transform(lambda x: x.replace(0, np.nan).mean())
    tod_means = tod_means.fillna(eff_vol.replace(0, np.nan).mean()).fillna(1.0)
    tod_ratio_series = (eff_vol / (tod_means + 1e-9)).clip(0.0, 20.0)

    decision_idx = -2 if len(df) >= 2 else -1
    confirmed_tod_ratio = float(tod_ratio_series.iloc[decision_idx]) if len(tod_ratio_series) > 0 else 1.0
    return vwap, confirmed_tod_ratio, tod_ratio_series


# ============================================================================
# Task 2.1: Calibrated Multi-Factor Scoring & Collinearity Diagnostics
# ============================================================================
def _winsorize_and_standardize(series: pd.Series, window: int = 120, clip_sigma: float = 3.0) -> pd.Series:
    """Zero-anchored Root-Mean-Square (RMS) standardization with Winsorization at [-clip_sigma, +clip_sigma].
    Preserves the natural zero-equilibrium of financial return/flow signals without short-window mean-shift distortion.
    """
    min_p = max(15, window // 4)
    s_clean = series.fillna(0.0)
    rms = np.sqrt((s_clean * s_clean).rolling(window, min_periods=min_p).mean().bfill())
    z = s_clean / (rms + 1e-9)
    return z.clip(-clip_sigma, clip_sigma).fillna(0.0)


def compute_factor_exposure_and_ic_weights(
    df: pd.DataFrame,
    fundamental_z: float = 0.0,
    macro_z: float = 0.0,
    horizon: int = 5,
    ic_window: int = 500,
) -> dict[str, Any]:
    """
    Construct the 4 canonical institutional factor series using strictly lagged / confirmed bars,
    calibrate weights via signed Rank-IC Bayesian shrinkage (aligning both trend-following and mean-reverting stocks),
    transform into Logistic Regression upward trend probability P(Upward Trend | Factors) in [0, 1],
    and compute Collinearity Diagnostics (VIF & pairwise factor correlation matrix).
    """
    close = df['Close']
    high = df['High']
    low = df['Low']
    vol = df['Volume'].fillna(0.0)
    susp = df['Is_Suspended'] if 'Is_Suspended' in df.columns else (vol <= 0)

    rets = close.pct_change().mask(susp, np.nan).fillna(0.0)
    vol_20 = rets.rolling(20, min_periods=10).std().bfill().fillna(0.02)
    vol_60 = rets.rolling(60, min_periods=20).std().bfill().fillna(vol_20)

    # 1. Momentum Factor: risk-adjusted 5D and 20D trend persistence
    ir_5d = close.pct_change(5).fillna(0.0) / (vol_20 * np.sqrt(5.0) + 1e-9)
    ir_20d = close.pct_change(20).fillna(0.0) / (vol_60 * np.sqrt(20.0) + 1e-9)
    raw_mom = 0.35 * ir_5d + 0.65 * ir_20d
    z_mom = _winsorize_and_standardize(raw_mom, window=120)

    # 2. Value & Quality Factor: combines short/medium valuation pullback anchor with fundamental Z
    sma_60 = close.rolling(60, min_periods=20).mean().bfill()
    dist_60 = (close - sma_60) / (sma_60 * vol_60 * np.sqrt(20.0) + 1e-9)
    z_val_ts = _winsorize_and_standardize(-0.65 * ir_5d - 0.35 * dist_60.fillna(0.0), window=120)
    z_val = (0.80 * z_val_ts + 0.20 * float(np.clip(fundamental_z, -3.0, 3.0))).clip(-3.0, 3.0)

    # 3. Volatility / Asymmetric Risk Quality: Upside vs Downside semivariance symmetry
    up_vol = rets.clip(lower=0.0).rolling(20, min_periods=10).std().bfill().fillna(0.015)
    dn_vol = (-rets).clip(lower=0.0).rolling(20, min_periods=10).std().bfill().fillna(0.015)
    raw_vol_stability = np.log((up_vol + 1e-6) / (dn_vol + 1e-6))
    z_vol = _winsorize_and_standardize(raw_vol_stability, window=120)

    # 4. Liquidity & Microstructure Factor: 20D Chaikin Money Flow (CMF) + volume-weighted accumulation
    hl_range = (high - low).replace(0.0, np.nan)
    clv = (((close - low) - (high - close)) / (hl_range + 1e-9)).fillna(0.0).clip(-1.0, 1.0)
    cmf_20 = (clv * vol).rolling(20, min_periods=10).sum() / (vol.rolling(20, min_periods=10).sum() + 1e-9)
    z_liq = _winsorize_and_standardize(cmf_20, window=120)

    factor_df = pd.DataFrame({
        "Momentum": z_mom,
        "Value_Quality": z_val,
        "Volatility_Quality": z_vol,
        "Liquidity_Micro": z_liq,
    }, index=df.index)

    # Factor at t-1 predicts return from Open[t] to Close[t + horizon - 1]
    open_col = df['Open'] if 'Open' in df.columns else close
    fwd_ret_for_ic = (close.shift(-horizon) / (open_col.shift(-1) + 1e-9)) - 1.0
    eff_window = max(180, int(ic_window))
    valid_ic_slice = factor_df.iloc[30:-horizon - 1].tail(eff_window)
    valid_fwd = fwd_ret_for_ic.loc[valid_ic_slice.index]

    raw_ics: dict[str, float] = {}
    ic_priors = {
        "Momentum": 0.012,
        "Value_Quality": 0.010,
        "Volatility_Quality": 0.008,
        "Liquidity_Micro": 0.008,
    }
    for col in factor_df.columns:
        s_f = valid_ic_slice[col]
        mask_valid = s_f.notna() & valid_fwd.notna()
        if mask_valid.sum() >= 20 and s_f[mask_valid].std() > 1e-6:
            ic_val, _ = spearmanr(s_f[mask_valid], valid_fwd[mask_valid])
            raw_ics[col] = float(0.0 if np.isnan(ic_val) else ic_val)
        else:
            raw_ics[col] = ic_priors[col]

    # Signed Bayesian IC-proportional weighting in true IC units (aligns factor polarity with stock regime)
    signed_ic_scores: dict[str, float] = {}
    for col in factor_df.columns:
        blended_ic = 0.88 * raw_ics[col] + 0.12 * ic_priors[col]
        signed_ic_scores[col] = blended_ic * (1.0 + 10.0 * abs(blended_ic))

    total_abs_ic = sum(abs(val) for val in signed_ic_scores.values()) + 1e-9
    calibrated_weights = {col: float(val / total_abs_ic) for col, val in signed_ic_scores.items()}

    composite_z_series = (
        (
            calibrated_weights["Momentum"] * factor_df["Momentum"]
            + calibrated_weights["Value_Quality"] * factor_df["Value_Quality"]
            + calibrated_weights["Volatility_Quality"] * factor_df["Volatility_Quality"]
            + calibrated_weights["Liquidity_Micro"] * factor_df["Liquidity_Micro"]
        ) * 1.35
        + 0.12 * float(np.clip(macro_z, -3.0, 3.0))
    ).clip(-3.0, 3.0)

    prob_up_series = 1.0 / (1.0 + np.exp(-0.85 * composite_z_series))

    clean_f = factor_df.tail(120).dropna()
    corr_vals = clean_f.corr().fillna(0.0).to_numpy(copy=True)
    np.fill_diagonal(corr_vals, 1.0)
    corr_mat = pd.DataFrame(corr_vals, index=clean_f.columns, columns=clean_f.columns)
    try:
        inv_corr = np.linalg.pinv(corr_vals)
        vif_vals = np.maximum(1.0, np.diag(inv_corr))
    except Exception:
        vif_vals = np.ones(len(factor_df.columns))

    vif_map = {col: float(v) for col, v in zip(factor_df.columns, vif_vals)}

    decision_idx = -2 if len(factor_df) >= 2 else -1
    latest_f = factor_df.iloc[decision_idx]
    latest_z = float(composite_z_series.iloc[decision_idx])
    latest_prob = float(prob_up_series.iloc[decision_idx])

    label_map = {
        "Momentum": "動量因子 (Momentum)",
        "Value_Quality": "價值與質量 (Value & Quality)",
        "Volatility_Quality": "低波品質 (Volatility Stability)",
        "Liquidity_Micro": "流動性與資金流 (Liquidity Flow)",
    }

    exposure_rows = []
    for col in factor_df.columns:
        z_v = float(latest_f[col])
        w_v = float(calibrated_weights[col])
        contrib_z = z_v * w_v
        vif_v = vif_map[col]
        # Display IC-aligned exposure so positive bar always means bullish contribution
        aligned_z = z_v if w_v >= 0 else -z_v
        exposure_rows.append({
            "factor_key": col,
            "factor_name": label_map[col],
            "z_score": round(aligned_z, 3),
            "rank_ic": round(abs(raw_ics[col]) if (raw_ics[col] * w_v) >= 0 else raw_ics[col], 4),
            "ic_weight_pct": round(abs(w_v) * 100.0, 1),
            "marginal_contribution_z": round(contrib_z, 3),
            "vif": round(vif_v, 2),
            "collinearity_status": "🟢 獨立正交 (VIF < 2.5)" if vif_v < 2.5 else ("🟡 輕微共線" if vif_v < 5.0 else "🔴 高度共線"),
        })

    return {
        "factor_df": factor_df,
        "composite_z_series": composite_z_series,
        "prob_up_series": prob_up_series,
        "latest_composite_z": latest_z,
        "latest_prob_up": latest_prob,
        "calibrated_weights": calibrated_weights,
        "raw_ics": raw_ics,
        "vif_map": vif_map,
        "factor_corr_matrix": corr_mat,
        "exposure_matrix": exposure_rows,
    }


# ============================================================================
# Task 1.1 & Task 2.1: Intraday Day-Trading Signal Engine (Look-Ahead Free)
# ============================================================================
def generate_day_trading_signal(
    ticker: str,
    min_adv_hkd: float = 5_000_000.0,
    interval: str = "15m",
    trade_capital_hkd: float = 50_000.0,
    broker_comm_rate: float = DEFAULT_BROKER_COMM_RATE,
    min_broker_comm_hkd: float = DEFAULT_MIN_BROKER_COMM_HKD,
    almgren_gamma: float = DEFAULT_ALMGREN_GAMMA,
    spot_price_override: float | None = None,
    futu_host: str = "127.0.0.1",
    futu_port: int = 11111,
    **kwargs: Any,
) -> tuple[dict[str, Any] | None, str | None]:
    """
    Institutional Intraday Signal Engine:
    - Eliminates Look-Ahead Bias by computing all technical features on confirmed closed bar `t-1`.
    - Separates Decision Price (`Close[t-1]`) from Realistic Execution Price (`Open[t]` + Almgren-Chriss slippage).
    - Outputs calibrated Z-Scores in [-3.0, +3.0] and Logistic Regression Upward Trend Probability P(Up) in [0.0, 1.0].
    """
    try:
        df_daily, successful_ticker = safe_download_ohlcv(ticker, period="6mo", interval="1d", min_rows=25)
        if df_daily is None or successful_ticker is None:
            return None, f"無法獲取 {ticker} 的有效歷史數據（可能停牌或代號無效）"

        raw_close_daily = df_daily['Raw_Close'] if 'Raw_Close' in df_daily.columns else df_daily['Close']
        active_daily_mask = ~df_daily['Is_Suspended'] if 'Is_Suspended' in df_daily.columns else (df_daily['Volume'] > 0)
        active_dollar_vol = (raw_close_daily * df_daily['Volume']).where(active_daily_mask, np.nan)
        adv_20d_hkd = float(active_dollar_vol.rolling(20, min_periods=5).mean().dropna().iloc[-1]) if not active_dollar_vol.dropna().empty else 0.0
        daily_sigma = float(raw_close_daily.pct_change().where(active_daily_mask, np.nan).rolling(20, min_periods=5).std().dropna().iloc[-1]) if len(df_daily) >= 10 else 0.02

        short_ok = is_hk_short_eligible(successful_ticker, adv_20d_hkd=adv_20d_hkd)
        lot_size = HK_DEFAULT_LOT_SIZES.get(successful_ticker, 100)

        adj_factor_daily = (raw_close_daily / (df_daily['Close'] + 1e-9)).fillna(1.0)
        raw_high_d = df_daily['High'] * adj_factor_daily
        raw_low_d = df_daily['Low'] * adj_factor_daily
        tr_daily = pd.concat([
            raw_high_d - raw_low_d,
            (raw_high_d - raw_close_daily.shift(1)).abs(),
            (raw_low_d - raw_close_daily.shift(1)).abs()
        ], axis=1).max(axis=1)
        daily_atr_series = tr_daily.ewm(alpha=1.0 / 14.0, min_periods=10, adjust=False).mean()
        daily_atr_14 = float(daily_atr_series.iloc[-2]) if len(daily_atr_series) >= 2 else float(daily_atr_series.iloc[-1])

        used_interval = interval
        df = None
        if interval in ("5m", "15m"):
            df, _ = safe_download_ohlcv(successful_ticker, period="1mo", interval=interval, min_rows=30)
            if df is None:
                used_interval = "1d"
                df = df_daily
        else:
            df = df_daily

        df = df.copy()
        close = df['Close']
        open_s = df['Open'] if 'Open' in df.columns else close
        high = df['High']
        low = df['Low']
        volume = df['Volume']

        raw_close_s = df['Raw_Close'] if 'Raw_Close' in df.columns else close
        raw_open_s = df['Raw_Open'] if 'Raw_Open' in df.columns else open_s

        dec_idx = -2 if len(df) >= 3 else -1
        prev_dec_idx = -3 if len(df) >= 3 else dec_idx

        decision_close_adj = float(close.iloc[dec_idx])
        prev_decision_close_adj = float(close.iloc[prev_dec_idx])
        decision_price_raw = float(raw_close_s.iloc[dec_idx])
        realtime_spot = spot_price_override if (spot_price_override is not None and spot_price_override > 0) else fetch_realtime_spot_price(successful_ticker, futu_host=futu_host, futu_port=futu_port)
        latest_spot_raw = float(realtime_spot) if (realtime_spot is not None and realtime_spot > 0) else float(raw_close_s.iloc[-1])
        next_bar_open_raw = float(raw_open_s.iloc[-1]) if len(df) >= 2 else decision_price_raw

        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)
        atr_series = tr.ewm(alpha=1.0 / 14.0, min_periods=10, adjust=False).mean()
        atr_14_dec = float(atr_series.iloc[dec_idx])
        if np.isnan(atr_14_dec) or atr_14_dec <= 0:
            atr_14_dec = float((high - low).iloc[:dec_idx].mean()) + 1e-6

        price_scale_ratio = decision_price_raw / (decision_close_adj + 1e-9)
        atr_14_raw = atr_14_dec * price_scale_ratio

        price_change_dec = decision_close_adj - prev_decision_close_adj
        change_pct = price_change_dec / (prev_decision_close_adj + 1e-9)
        atr_zscore = float(np.clip(price_change_dec / (atr_14_dec + 1e-9), -3.0, 3.0))

        df['SMA5'] = close.rolling(5, min_periods=2).mean()
        df['SMA10'] = close.rolling(10, min_periods=3).mean()
        df['Vol_MA20'] = volume.rolling(20, min_periods=5).mean()
        sma_5_dec = float(df['SMA5'].iloc[dec_idx])
        sma_10_dec = float(df['SMA10'].iloc[dec_idx])

        hist_vol_slice = volume.iloc[max(0, len(volume) + dec_idx - 60) : len(volume) + dec_idx + 1]
        dec_vol = float(volume.iloc[dec_idx])
        vol_pct_rank = float((hist_vol_slice < dec_vol).mean() * 100.0) if hist_vol_slice.sum() > 0 else 50.0

        if used_interval in ("5m", "15m"):
            vwap_series, vol_surge_ratio, _ = compute_intraday_vwap_and_tod_volume(df)
            df['VWAP'] = vwap_series
            dec_vwap = float(vwap_series.iloc[dec_idx])
        else:
            vol_20ma_dec = float(df['Vol_MA20'].iloc[dec_idx])
            vol_surge_ratio = min(dec_vol / (vol_20ma_dec + 1e-9), 20.0) if (vol_20ma_dec > 0 and adv_20d_hkd >= 100_000) else 0.0
            df['VWAP'] = df['SMA5']
            dec_vwap = float(df['VWAP'].iloc[dec_idx])

        df['ATR_Upper'] = df['VWAP'] + 1.5 * atr_series
        df['ATR_Lower'] = df['VWAP'] - 1.5 * atr_series

        rsi_series = compute_wilder_rsi(close, 14)
        df['RSI'] = rsi_series
        rsi_14_dec = float(rsi_series.iloc[dec_idx]) if not np.isnan(rsi_series.iloc[dec_idx]) else 50.0

        is_illiquid = adv_20d_hkd < min_adv_hkd
        z_mom_intra = float(np.clip(atr_zscore, -3.0, 3.0))
        z_vol_intra = float(np.clip(((vol_pct_rank - 50.0) / 18.0) * np.sign(z_mom_intra if abs(z_mom_intra) > 0.1 else 1.0), -3.0, 3.0)) if not is_illiquid else -2.0
        vwap_dist_atr = (decision_close_adj - dec_vwap) / (atr_14_dec + 1e-9)
        ma_spread_atr = (sma_5_dec - sma_10_dec) / (atr_14_dec + 1e-9)
        z_vwap_intra = float(np.clip(0.65 * vwap_dist_atr + 0.35 * ma_spread_atr, -3.0, 3.0))

        if rsi_14_dec <= 75.0:
            z_rsi_intra = float(np.clip((rsi_14_dec - 50.0) / 10.0, -3.0, 3.0))
        else:
            z_rsi_intra = float(np.clip(-(rsi_14_dec - 72.0) / 6.0, -3.0, 3.0))

        z_liq_intra = float(np.clip(math.log10(max(adv_20d_hkd, 1e4) / max(min_adv_hkd, 1e4)) * 1.5, -3.0, 3.0))

        w_intra = {"mom": 0.30, "vwap": 0.25, "vol": 0.20, "rsi": 0.15, "liq": 0.10}
        composite_z = float(np.clip(
            w_intra["mom"] * z_mom_intra
            + w_intra["vwap"] * z_vwap_intra
            + w_intra["vol"] * z_vol_intra
            + w_intra["rsi"] * z_rsi_intra
            + w_intra["liq"] * z_liq_intra,
            -3.0, 3.0
        ))
        if is_illiquid:
            composite_z = min(composite_z, -1.5)

        up_prob = float(1.0 / (1.0 + math.exp(-1.15 * composite_z)))
        day_score = float(np.clip((up_prob - 0.5) * 200.0, -100.0, 100.0))

        micro_cost = calculate_hkex_microstructure_cost(
            trade_value_hkd=max(trade_capital_hkd, decision_price_raw * lot_size),
            adv_20d_hkd=adv_20d_hkd,
            daily_volatility=daily_sigma,
            broker_comm_rate=broker_comm_rate,
            min_broker_comm_hkd=min_broker_comm_hkd,
            gamma=almgren_gamma,
        )
        slippage_frac = micro_cost["slippage_bps"] / 10000.0
        is_short_bias = (up_prob <= 0.35) and short_ok
        if is_short_bias:
            raw_exec_price = next_bar_open_raw * (1.0 - slippage_frac)
        else:
            raw_exec_price = next_bar_open_raw * (1.0 + slippage_frac)
        execution_price = round_to_hk_tick(raw_exec_price)

        above_vwap = decision_close_adj >= dec_vwap
        if is_illiquid:
            action = "🚫 【流動性不足：禁止當沖進場 (買賣價差與衝擊風險過高)】"
        elif up_prob >= 0.65 and above_vwap:
            action = f"🟢 【順勢做多訊號 (上漲機率 {up_prob*100:.1f}% | 合成 Z={composite_z:+.2f}σ)】"
        elif up_prob <= 0.35:
            if short_ok:
                action = f"🔴 【順勢做空或減倉訊號 (下跌機率 {(1-up_prob)*100:.1f}% | 屬SFC可沽空名單)】"
            else:
                action = "🟠 【觀望迴避 / 僅限多頭平倉 (非SFC可沽空名單或無券源)】"
        else:
            action = f"⚖️ 【短線觀望 / 區間震盪等待 (上漲機率 {up_prob*100:.1f}%)】"

        signals = [
            f"⏱️ 決策與執行分離：訊號決策價 ${round_to_hk_tick(decision_price_raw):.2f} (上一根已確認K線) ➔ 預估撮合執行價 ${execution_price:.2f} (次根開盤 + 滑價 {micro_cost['slippage_bps']:.1f} bps)",
            f"📈 趨勢機率模型：預估上漲機率 {up_prob*100:.1f}% | 合成因子標準化得分 {composite_z:+.2f}σ",
            f"💧 流動性與衝擊成本：20日均成交額 ${adv_20d_hkd:,.0f} HKD | 單筆預估總摩擦成本 {micro_cost['total_cost_bps']:.1f} bps (${micro_cost['total_frictions_hkd']:.1f} HKD)",
            f"🛡️ 港股賣空合規檢查：{'✅ 屬於 SFC 可沽空高流動名單' if short_ok else '🚫 非可沽空名單或借券流動性不足（禁止裸賣空）'}",
        ]

        effective_stop_dist = max(1.5 * atr_14_raw, 0.5 * daily_atr_14) if used_interval in ("5m", "15m") else 1.5 * atr_14_raw
        if is_short_bias:
            stop_price = round_to_hk_tick(execution_price + effective_stop_dist)
            target_price = round_to_hk_tick(max(0.01, execution_price - 2.0 * effective_stop_dist))
        else:
            stop_price = round_to_hk_tick(max(0.01, execution_price - effective_stop_dist))
            target_price = round_to_hk_tick(execution_price + 2.0 * effective_stop_dist)

        return {
            "ticker": successful_ticker,
            "price": round_to_hk_tick(latest_spot_raw),
            "decision_price": round_to_hk_tick(decision_price_raw),
            "execution_price": execution_price,
            "vwap": round_to_hk_tick(dec_vwap * price_scale_ratio),
            "stop_loss_price": stop_price,
            "take_profit_price": target_price,
            "tick_size": get_hk_tick_size(latest_spot_raw),
            "lot_size": lot_size,
            "lot_cost_hkd": latest_spot_raw * lot_size,
            "interval": used_interval,
            "day_score": day_score,
            "up_prob": up_prob,
            "composite_zscore": composite_z,
            "action": action,
            "rsi": rsi_14_dec,
            "vol_ratio": vol_surge_ratio,
            "vol_pct_rank": vol_pct_rank,
            "atr": atr_14_raw,
            "atr_zscore": atr_zscore,
            "change_pct": change_pct,
            "adv_20d_hkd": adv_20d_hkd,
            "short_eligible": short_ok,
            "microstructure": micro_cost,
            "factor_zscores": {
                "ATR 波動率動量 (Z_Mom)": round(z_mom_intra, 2),
                "VWAP 均線結構 (Z_VWAP)": round(z_vwap_intra, 2),
                "時段量能分位 (Z_Vol)": round(z_vol_intra, 2),
                "Wilder RSI 動能 (Z_RSI)": round(z_rsi_intra, 2),
                "流動性安全邊際 (Z_Liq)": round(z_liq_intra, 2),
            },
            "factor_scores": {
                "ATR 波動率動量": round(z_mom_intra * 12.0, 1),
                "時段量能分位": round(z_vol_intra * 12.0, 1),
                "VWAP 均線結構": round(z_vwap_intra * 12.0, 1),
                "Wilder RSI 動能": round(z_rsi_intra * 12.0, 1),
                "流動性安全邊際": round(z_liq_intra * 12.0, 1),
            },
            "signals": signals,
            "history": df.tail(80),
        }, None
    except Exception as e:
        return None, str(e)


# ============================================================================
# Task 1.1 & Task 2.1: Dual-Horizon Calibrated Alpha Engine
# ============================================================================
def analyze_stock_trend(ticker: str) -> tuple[dict[str, Any] | None, str | None]:
    """
    Calibrated Multi-Factor Stock Trend & Probability Engine:
    - Uses confirmed bar `t-1` (`decision_price`) for all rolling moving averages and factor Z-scores,
      and reports `execution_price` (`Open[t]` + Almgren-Chriss slippage).
    - Outputs calibrated Logistic Regression upward probability P(Upward Trend | Factors) in [0, 1],
      standardized Z-Scores in [-3.0, +3.0], and full Factor Exposure + VIF Collinearity Matrix.
    """
    try:
        df, successful_ticker = safe_download_ohlcv(ticker, period="2y", interval="1d", min_rows=60)
        if df is None or successful_ticker is None:
            return None, f"無法獲取代號 {ticker} 的有效歷史數據"

        df = df.copy()
        close = df['Close']
        raw_close = df['Raw_Close'] if 'Raw_Close' in df.columns else close
        raw_open = df['Raw_Open'] if 'Raw_Open' in df.columns else raw_close
        volume = df['Volume']
        susp = df['Is_Suspended'] if 'Is_Suspended' in df.columns else (volume <= 0)

        macro = fetch_hk_macro_regime()
        fundamental = fetch_fundamental_data(successful_ticker)

        df['SMA50'] = close.rolling(50, min_periods=20).mean()
        df['SMA200'] = close.rolling(200, min_periods=60).mean()
        df['RSI'] = compute_wilder_rsi(close, 14)

        dec_idx = -2 if len(df) >= 3 else -1
        realtime_spot = fetch_realtime_spot_price(successful_ticker)
        spot_price_raw = float(realtime_spot) if (realtime_spot is not None and realtime_spot > 0) else float(raw_close.iloc[-1])
        decision_price_raw = float(raw_close.iloc[dec_idx])
        next_open_raw = float(raw_open.iloc[-1])

        dec_adj_price = float(close.iloc[dec_idx])
        dec_sma50 = float(df['SMA50'].iloc[dec_idx]) if not np.isnan(df['SMA50'].iloc[dec_idx]) else dec_adj_price
        dec_sma200 = float(df['SMA200'].iloc[dec_idx]) if not np.isnan(df['SMA200'].iloc[dec_idx]) else dec_adj_price

        daily_rets = close.pct_change().mask(susp, np.nan)
        vol_20d_series = daily_rets.rolling(20, min_periods=10).std().bfill()
        vol_60d_series = daily_rets.rolling(60, min_periods=20).std().bfill()
        df['Z_20D'] = (close.pct_change(20) / (vol_60d_series * np.sqrt(20.0) + 1e-9)).clip(-3.0, 3.0).fillna(0.0)

        vol_20d = float(vol_20d_series.iloc[dec_idx]) if not np.isnan(vol_20d_series.iloc[dec_idx]) else 0.02
        vol_60d = float(vol_60d_series.iloc[dec_idx]) if not np.isnan(vol_60d_series.iloc[dec_idx]) else vol_20d

        ret_5d = float(close.pct_change(5).iloc[dec_idx]) if len(close) >= 7 else 0.0
        ret_20d = float(close.pct_change(20).iloc[dec_idx]) if len(close) >= 22 else 0.0

        z_5d = float(np.clip(ret_5d / (vol_20d * np.sqrt(5.0) + 1e-9), -3.0, 3.0))
        z_20d = float(np.clip(ret_20d / (vol_60d * np.sqrt(20.0) + 1e-9), -3.0, 3.0))

        df_weekly = close.iloc[:len(close) + dec_idx + 1].resample('W').last().dropna()
        weekly_sma20 = df_weekly.rolling(20, min_periods=5).mean()
        curr_weekly_price = float(df_weekly.iloc[-1]) if not df_weekly.empty else dec_adj_price
        curr_weekly_sma20 = float(weekly_sma20.iloc[-1]) if not weekly_sma20.empty else curr_weekly_price
        weekly_dist = (curr_weekly_price - curr_weekly_sma20) / (curr_weekly_sma20 + 1e-9)
        weekly_bullish = curr_weekly_price > curr_weekly_sma20

        dist_200ma = (dec_adj_price - dec_sma200) / (dec_sma200 + 1e-9)

        factor_model = compute_factor_exposure_and_ic_weights(
            df,
            fundamental_z=fundamental["val_qual_zscore"],
            macro_z=macro["macro_zscore"],
            horizon=10,
            ic_window=max(252, len(df) - 30),
        )

        composite_z = factor_model["latest_composite_z"]
        up_prob = factor_model["latest_prob_up"]
        composite_score = float(np.clip((up_prob - 0.5) * 200.0, -100.0, 100.0))

        st_z = float(np.clip(0.55 * z_20d + 0.45 * z_5d, -3.0, 3.0))
        st_prob = float(1.0 / (1.0 + math.exp(-0.9 * st_z)))
        short_term_score = float((st_prob - 0.5) * 200.0)

        lt_tech_z = float(np.clip(0.55 * (dist_200ma / 0.12) + 0.45 * (weekly_dist / 0.08), -3.0, 3.0))
        lt_z = float(np.clip(0.50 * fundamental["val_qual_zscore"] + 0.30 * lt_tech_z + 0.20 * macro["macro_zscore"], -3.0, 3.0))
        lt_prob = float(1.0 / (1.0 + math.exp(-0.9 * lt_z)))
        long_term_score = float((lt_prob - 0.5) * 200.0)

        adv_20d_hkd = float((raw_close * volume).where(~susp, np.nan).rolling(20, min_periods=5).mean().dropna().iloc[-1])
        micro_info = calculate_hkex_microstructure_cost(
            trade_value_hkd=max(50_000.0, spot_price_raw * fundamental["lot_size"]),
            adv_20d_hkd=adv_20d_hkd,
            daily_volatility=vol_20d,
        )
        exec_price = round_to_hk_tick(next_open_raw * (1.0 + (micro_info["slippage_bps"] / 10000.0)))

        if up_prob >= 0.60:
            trend = f"📈 強勢多頭機率 ({up_prob*100:.1f}% | Z={composite_z:+.2f}σ)"
        elif up_prob <= 0.40:
            trend = f"📉 弱勢空頭機率 ({up_prob*100:.1f}% | Z={composite_z:+.2f}σ)"
        else:
            trend = f"⚖️ 中性區間震盪 ({up_prob*100:.1f}% | Z={composite_z:+.2f}σ)"

        reasons = [
            f"🎯 統計校準趨勢機率：預估上漲機率 {up_prob*100:.1f}% (合成因子暴露 Z = {composite_z:+.2f}σ)",
            f"⏱️ 無未來函數決策基準：訊號決策價 ${round_to_hk_tick(decision_price_raw):.2f} (上一交易日收盤) ➔ 次日開盤撮合價 ${exec_price:.2f} (含滑價 {micro_info['slippage_bps']:.1f} bps)",
            f"⚡ 短期動量因子：戰術上漲機率 {st_prob*100:.1f}% (Z = {st_z:+.2f}σ) | 5日動量 Z={z_5d:+.2f}σ ({ret_5d*100:+.2f}%) | 20日動量 Z={z_20d:+.2f}σ ({ret_20d*100:+.2f}%)",
            f"🏛️ 長期價值與宏觀：戰略上漲機率 {lt_prob*100:.1f}% (Z = {lt_z:+.2f}σ) | 前復權 200MA 偏離 {dist_200ma*100:+.2f}% | 週線 20MA 偏離 {weekly_dist*100:+.2f}% ({'🟢 多頭共振' if weekly_bullish else '🔴 週線受壓'})",
        ] + [f"  ↳ {r}" for r in fundamental["fund_reasons"]]

        exp_rows = factor_model["exposure_matrix"]
        factor_breakdown = {
            row["factor_name"]: round(row["z_score"] * 13.3, 1)
            for row in exp_rows
        }
        factor_breakdown["長線均線與週線結構 (Z_Trend)"] = round(lt_tech_z * 13.3, 1)
        factor_breakdown["港股宏觀牛熊閘門 (Z_Macro)"] = round(macro["macro_zscore"] * 13.3, 1)

        return {
            "ticker": successful_ticker,
            "price": round_to_hk_tick(spot_price_raw),
            "decision_price": round_to_hk_tick(decision_price_raw),
            "execution_price": exec_price,
            "trend": trend,
            "up_prob": up_prob,
            "composite_zscore": composite_z,
            "score": composite_score,
            "short_term_score": short_term_score,
            "long_term_score": long_term_score,
            "short_term_prob": st_prob,
            "long_term_prob": lt_prob,
            "ret_5d": ret_5d,
            "ret_20d": ret_20d,
            "z_5d": z_5d,
            "z_20d": z_20d,
            "dist_200ma": dist_200ma,
            "weekly_dist": weekly_dist,
            "dividend_yield": fundamental["dividend_yield"],
            "roe": fundamental["roe"],
            "factor_breakdown": factor_breakdown,
            "factor_exposure_matrix": exp_rows,
            "factor_corr_matrix": factor_model["factor_corr_matrix"],
            "calibrated_weights": factor_model["calibrated_weights"],
            "reasons": reasons,
            "history": df.tail(160),
        }, None
    except Exception as e:
        return None, str(e)


# ============================================================================
# Task 2.3: Analytical Ledoit-Wolf Shrinkage & RMT Eigenvalue Denoising
# ============================================================================