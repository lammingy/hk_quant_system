"""
engine/risk.py - Ledoit-Wolf 壓縮、RMT 矩陣去噪、HRP / Risk Parity 與 L1 換手約束
===================================================================================
Provides analytical Ledoit-Wolf covariance shrinkage, Random Matrix Theory (RMT) Marcenko-Pastur
eigenvalue clipping, Hierarchical Risk Parity (HRP), Equal Risk Contribution (Risk Parity),
L1 turnover constraints, Almgren-Chriss execution orders, and portfolio health diagnostics.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.optimize import minimize
from scipy.spatial.distance import squareform
from scipy.stats import kurtosis, norm, skew

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
        load_portfolio_config,
        normalize_hk_ticker,
        round_to_hk_tick,
    )
    from hk_quant_system.data.futu_gateway import check_futu_opend_connection, fetch_realtime_spot_price
    from hk_quant_system.data.loader import (
        compute_active_daily_returns,
        fetch_fundamental_data,
        fetch_hk_macro_regime,
        log_snapshot_to_db,
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
        load_portfolio_config,
        normalize_hk_ticker,
        round_to_hk_tick,
    )
    from data.futu_gateway import check_futu_opend_connection, fetch_realtime_spot_price
    from data.loader import (
        compute_active_daily_returns,
        fetch_fundamental_data,
        fetch_hk_macro_regime,
        log_snapshot_to_db,
        safe_download_ohlcv,
    )

def compute_ledoit_wolf_cov(returns_df: pd.DataFrame) -> tuple[np.ndarray, float]:
    """
    Analytical Ledoit-Wolf (2004) Shrinkage Covariance estimator towards constant-correlation target.
    """
    clean_X = returns_df.dropna(how="all").fillna(0.0).values
    T, N = clean_X.shape
    if N == 1:
        var_1 = float(np.nanvar(returns_df.values, ddof=1)) * 252.0
        return np.array([[max(var_1, 1e-4)]]), 0.0
    if T < 5:
        eye_cov = np.eye(N) * 0.04
        return eye_cov, 1.0

    X = clean_X - np.mean(clean_X, axis=0, keepdims=True)
    sample_cov = (X.T @ X) / float(T)

    var = np.maximum(np.diag(sample_cov), 1e-8)
    std = np.sqrt(var)
    outer_std = np.outer(std, std)
    corr = np.clip(sample_cov / (outer_std + 1e-12), -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)
    r_bar = (np.sum(corr) - N) / max(1.0, float(N * (N - 1)))
    target_F = r_bar * outer_std
    np.fill_diagonal(target_F, var)

    X2 = X ** 2
    pi_mat = (X2.T @ X2) / float(T) - (sample_cov ** 2)
    pi_hat = float(np.sum(pi_mat))
    gamma_hat = float(np.sum((sample_cov - target_F) ** 2))

    kappa = (pi_hat / (gamma_hat + 1e-12)) / float(T)
    shrinkage_delta = float(np.clip(kappa, 0.05, 1.0))

    shrunk_cov = ((1.0 - shrinkage_delta) * sample_cov + shrinkage_delta * target_F) * 252.0
    min_eig = np.min(np.linalg.eigvalsh(shrunk_cov))
    if min_eig < 1e-6:
        shrunk_cov += np.eye(N) * (1e-5 - min_eig)
    return shrunk_cov, shrinkage_delta


def compute_rmt_denoised_cov(returns_df: pd.DataFrame) -> tuple[np.ndarray, float]:
    """
    Random Matrix Theory (RMT) Marcenko-Pastur Eigenvalue Clipping Covariance Denoising.
    """
    lw_cov, lw_delta = compute_ledoit_wolf_cov(returns_df)
    T, N = returns_df.shape
    if N <= 2 or T < 10:
        return lw_cov, lw_delta

    std = np.sqrt(np.maximum(np.diag(lw_cov), 1e-8))
    outer_std = np.outer(std, std)
    corr = np.clip(lw_cov / (outer_std + 1e-12), -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)

    eigvals, eigvecs = np.linalg.eigh(corr)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    q = float(N) / float(max(T, N + 1))
    lambda_plus = (1.0 + math.sqrt(q)) ** 2

    noise_mask = eigvals <= lambda_plus
    if np.any(noise_mask) and not np.all(noise_mask):
        eigvals_denoised = eigvals.copy()
        eigvals_denoised[noise_mask] = np.mean(eigvals[noise_mask])
    else:
        eigvals_denoised = np.maximum(eigvals, 1e-4)

    corr_denoised = eigvecs @ np.diag(eigvals_denoised) @ eigvecs.T
    d_scale = np.sqrt(np.maximum(np.diag(corr_denoised), 1e-8))
    corr_denoised = corr_denoised / np.outer(d_scale, d_scale)
    np.fill_diagonal(corr_denoised, 1.0)

    cov_denoised = corr_denoised * outer_std
    return cov_denoised, lw_delta


def _compute_hrp_weights(cov_matrix: np.ndarray, corr_matrix: np.ndarray, cols: list[str]) -> dict[str, float]:
    n = len(cols)
    if n == 1:
        return {cols[0]: 1.0}

    corr_clean = np.clip(np.nan_to_num(corr_matrix, nan=0.0), -1.0, 1.0)
    np.fill_diagonal(corr_clean, 1.0)
    dist = np.sqrt(np.clip(0.5 * (1.0 - corr_clean), 0.0, 1.0))
    np.fill_diagonal(dist, 0.0)

    condensed_dist = squareform(dist, checks=False)
    link = linkage(condensed_dist, method='single')
    sort_ix = list(leaves_list(link))

    weights = pd.Series(1.0, index=sort_ix)
    clustered_alphas = [sort_ix]

    def get_cluster_var(cov: np.ndarray, cluster_items: list[int]) -> float:
        cov_slice = cov[np.ix_(cluster_items, cluster_items)]
        ivp = 1.0 / (np.diag(cov_slice) + 1e-9)
        ivp /= np.sum(ivp)
        return float(np.dot(ivp.T, np.dot(cov_slice, ivp)))

    while len(clustered_alphas) > 0:
        clustered_alphas = [
            cluster[start:end]
            for cluster in clustered_alphas
            for start, end in ((0, len(cluster) // 2), (len(cluster) // 2, len(cluster)))
            if len(cluster) > 1
        ]
        for sub_idx in range(0, len(clustered_alphas), 2):
            left_cluster = clustered_alphas[sub_idx]
            right_cluster = clustered_alphas[sub_idx + 1]
            left_var = get_cluster_var(cov_matrix, left_cluster)
            right_var = get_cluster_var(cov_matrix, right_cluster)
            alloc_factor = 1.0 - left_var / (left_var + right_var + 1e-12)
            weights[left_cluster] *= alloc_factor
            weights[right_cluster] *= (1.0 - alloc_factor)

    weights = weights / weights.sum()
    return {cols[idx]: float(weights.loc[idx]) for idx in range(n)}


def _compute_risk_parity_weights(cov_matrix: np.ndarray, cols: list[str], max_single_weight: float) -> dict[str, float]:
    n = len(cols)
    target_rc = np.full(n, 1.0 / n)

    def risk_parity_obj(weights: np.ndarray) -> float:
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) + 1e-12
        mrc = np.dot(cov_matrix, weights) / port_vol
        rc = weights * mrc
        rc_pct = rc / port_vol
        return float(np.sum((rc_pct - target_rc) ** 2))

    min_w = min(0.05, 0.50 / n)
    max_w = max(float(max_single_weight), min(1.0, 1.35 / n))
    bounds = tuple((min_w, max_w) for _ in range(n))
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    ivp = 1.0 / (np.sqrt(np.diag(cov_matrix)) + 1e-9)
    init_guess = ivp / np.sum(ivp)

    res = minimize(risk_parity_obj, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    if res.success and not np.isnan(res.x).any():
        w = np.clip(res.x, min_w, max_w)
        w /= np.sum(w)
        return {col: float(val) for col, val in zip(cols, w)}
    return {col: float(val) for col, val in zip(cols, init_guess)}


def optimize_portfolio_weights(
    daily_rets: pd.DataFrame,
    max_single_weight: float = 0.35,
    shrink_alpha: float = 0.20,
    method: str = "hrp",
    cov_estimator: str = "ledoit_wolf",
    prev_weights: dict[str, float] | np.ndarray | None = None,
    max_turnover: float | None = 0.40,
) -> dict[str, float]:
    """
    Robust Portfolio Optimization with Ledoit-Wolf / RMT denoising and L1 Turnover Constraints.
    """
    cols = list(daily_rets.columns)
    num_assets = len(cols)
    if num_assets == 0:
        return {}
    if num_assets == 1:
        return {cols[0]: 1.0}

    halt_ratio = (daily_rets.isna() | (daily_rets.abs() < 1e-9)).mean(axis=0).values
    illiquidity_penalty = 1.0 / np.clip(1.0 - halt_ratio, 0.30, 1.0)

    est_key = (cov_estimator or "ledoit_wolf").lower()
    if est_key == "rmt":
        base_cov, _ = compute_rmt_denoised_cov(daily_rets)
    else:
        base_cov, _ = compute_ledoit_wolf_cov(daily_rets)

    cov_matrix = base_cov * np.outer(illiquidity_penalty, illiquidity_penalty)
    std_vec = np.sqrt(np.maximum(np.diag(cov_matrix), 1e-8))
    corr_matrix = np.clip(cov_matrix / (np.outer(std_vec, std_vec) + 1e-12), -1.0, 1.0)
    np.fill_diagonal(corr_matrix, 1.0)

    min_w = min(0.05, 0.50 / num_assets)
    max_w = max(float(max_single_weight), min(1.0, 1.35 / num_assets))

    method_key = (method or "hrp").lower()
    if method_key == "hrp":
        raw_w_dict = _compute_hrp_weights(cov_matrix, corr_matrix, cols)
    elif method_key == "risk_parity":
        raw_w_dict = _compute_risk_parity_weights(cov_matrix, cols, max_w)
    else:
        def portfolio_volatility(weights: np.ndarray) -> float:
            return float(np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))))

        bounds = tuple((min_w, max_w) for _ in range(num_assets))
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
        init_guess = np.full(num_assets, 1.0 / num_assets)
        result = minimize(
            portfolio_volatility, init_guess, method='SLSQP', bounds=bounds, constraints=constraints
        )
        if result.success and not np.isnan(result.x).any():
            raw_w_dict = {col: float(w) for col, w in zip(cols, result.x)}
        else:
            raw_w_dict = {col: 1.0 / num_assets for col in cols}

    w_arr = np.array([raw_w_dict[c] for c in cols], dtype=float)
    for _ in range(15):
        over_mask = w_arr > max_w
        under_mask = w_arr < min_w
        if not np.any(over_mask) and not np.any(under_mask):
            break
        w_arr = np.clip(w_arr, min_w, max_w)
        free_mask = ~(over_mask | under_mask)
        rem_needed = 1.0 - np.sum(w_arr)
        if np.any(free_mask) and np.sum(w_arr[free_mask]) > 0:
            w_arr[free_mask] += rem_needed * (w_arr[free_mask] / np.sum(w_arr[free_mask]))
        else:
            w_arr /= np.sum(w_arr)

    w_arr = np.clip(w_arr, min_w, max_w)
    w_arr /= np.sum(w_arr)

    if prev_weights is not None and max_turnover is not None and float(max_turnover) > 0.0:
        if isinstance(prev_weights, dict):
            prev_arr = np.array([float(prev_weights.get(c, 0.0)) for c in cols], dtype=float)
        else:
            prev_arr = np.asarray(prev_weights, dtype=float)
        if len(prev_arr) == num_assets and np.sum(prev_arr) > 0.5:
            prev_arr = prev_arr / np.sum(prev_arr)
            curr_turnover = float(np.sum(np.abs(w_arr - prev_arr)))
            max_to = float(max_turnover)
            if curr_turnover > max_to + 1e-6:
                blend_alpha = max_to / curr_turnover
                w_arr = prev_arr + blend_alpha * (w_arr - prev_arr)
                w_arr = np.clip(w_arr, min_w, max_w)
                w_arr /= np.sum(w_arr)

    return {col: float(w) for col, w in zip(cols, w_arr)}


def compute_deflated_sharpe_ratio(
    daily_rets: pd.Series,
    rf_rate: float = 0.04,
    num_trials: int = 5
) -> dict[str, Any]:
    """
    Compute Probabilistic & Deflated Sharpe Ratio (PSR / DSR) per Bailey & Lopez de Prado (2014).
    """
    clean_rets = daily_rets.dropna()
    T = len(clean_rets)
    if T < 20:
        return {"dsr_prob": 0.5, "p_value": 0.5, "sr_benchmark_ann": 0.0, "is_significant": False}

    rf_daily = rf_rate / 252.0
    excess = clean_rets - rf_daily
    std_d = float(excess.std())
    if std_d <= 1e-12:
        return {"dsr_prob": 0.0, "p_value": 1.0, "sr_benchmark_ann": 0.0, "is_significant": False}

    nav_end = float((1.0 + clean_rets).prod())
    cagr = (nav_end ** (252.0 / T) - 1.0) if nav_end > 0 else -1.0
    sr_daily = float(((cagr - rf_rate) / 252.0) / std_d)

    sk = float(skew(excess, bias=False))
    ku = float(kurtosis(excess, fisher=False, bias=False))

    K = max(1, int(num_trials))
    if K > 1:
        euler_gamma = 0.5772156649
        z1 = norm.ppf(1.0 - 1.0 / K)
        z2 = norm.ppf(1.0 - (1.0 / K) * np.exp(-1.0))
        sr0_daily = np.sqrt(1.0 / max(1, T - 1)) * ((1.0 - euler_gamma) * z1 + euler_gamma * z2)
    else:
        sr0_daily = 0.0

    denom_var = 1.0 - sk * sr_daily + ((ku - 1.0) / 4.0) * (sr_daily ** 2)
    denom = np.sqrt(max(1e-9, denom_var / max(1, T - 1)))
    z_stat = (sr_daily - sr0_daily) / denom
    dsr_prob = float(norm.cdf(z_stat))
    p_val = float(1.0 - dsr_prob)

    return {
        "dsr_prob": dsr_prob,
        "p_value": p_val,
        "sr_benchmark_ann": float(sr0_daily * np.sqrt(252.0)),
        "skewness": sk,
        "kurtosis": ku,
        "is_significant": dsr_prob >= 0.95,
    }


def generate_execution_orders(
    portfolio_items: list[dict[str, Any]],
    total_capital: float,
    optimal_weights: dict[str, float],
    latest_prices: pd.Series,
    adv_map: dict[str, float] | None = None,
    vol_map: dict[str, float] | None = None,
    min_trade_hkd: float = 8000.0,
    broker_comm_rate: float = DEFAULT_BROKER_COMM_RATE,
    min_broker_comm_hkd: float = DEFAULT_MIN_BROKER_COMM_HKD,
    almgren_gamma: float = DEFAULT_ALMGREN_GAMMA,
) -> list[dict[str, Any]]:
    """
    Generate realistic HKEX execution orders with statutory fee breakdown and Almgren-Chriss slippage.
    """
    orders = []

    for item in portfolio_items:
        t = item['ticker']
        if t not in latest_prices:
            continue
        current_shares = int(item.get('shares', 0))
        price = round_to_hk_tick(float(latest_prices[t]))
        lot_size = int(item.get('lot_size') or HK_DEFAULT_LOT_SIZES.get(t, 100))

        target_w = float(optimal_weights.get(t, 0.0))
        target_val = total_capital * target_w
        target_lots = math.floor((target_val / (price + 1e-9)) / lot_size)
        target_shares = target_lots * lot_size

        current_lots = current_shares // lot_size
        odd_shares = current_shares % lot_size
        shares_diff = target_shares - current_shares
        lots_diff = target_lots - current_lots

        trade_val = abs(shares_diff) * price
        adv_hkd = float(adv_map.get(t, 50_000_000.0)) if adv_map else 50_000_000.0
        sigma_d = float(vol_map.get(t, 0.02)) if vol_map else 0.02

        if 0 < trade_val < min_trade_hkd:
            shares_diff = 0
            lots_diff = 0
            target_shares = current_shares
            trade_val = 0.0
            cost_info = calculate_hkex_microstructure_cost(0.0)
            action = f"⚖️ 維持現狀 (差額低於 ${min_trade_hkd:,.0f} 經濟調倉門檻)"
        else:
            cost_info = calculate_hkex_microstructure_cost(
                trade_value_hkd=trade_val,
                adv_20d_hkd=adv_hkd,
                daily_volatility=sigma_d,
                broker_comm_rate=broker_comm_rate,
                min_broker_comm_hkd=min_broker_comm_hkd,
                gamma=almgren_gamma,
            )
            liq_note = (
                f" [⚠️ 佔ADV {cost_info['participation_rate']*100:.1f}% > 5%：流動性懲罰啟動，建議 TWAP 拆分為 {cost_info['recommended_twap_slices']} 單]"
                if cost_info["requires_twap"]
                else ""
            )

            if shares_diff > 0:
                action = f"🟢 買入 (BUY) {shares_diff:,} 股 ({abs(lots_diff)} 手){liq_note}"
            elif shares_diff < 0:
                action = f"🔴 賣出 (SELL) {abs(shares_diff):,} 股 ({abs(lots_diff)} 手){liq_note}"
            else:
                action = "⚖️ 維持現狀 (HOLD)"

        if odd_shares > 0:
            action += f" [含碎股 {odd_shares} 股]"

        orders.append({
            "ticker": t,
            "name": item.get("name", t),
            "lot_size": lot_size,
            "price": price,
            "target_weight": target_w,
            "current_shares": current_shares,
            "target_shares": target_shares,
            "action": action,
            "estimated_turnover": trade_val,
            "estimated_stamp_duty": cost_info["total_statutory_fees_hkd"],
            "statutory_fees_hkd": cost_info["total_statutory_fees_hkd"],
            "slippage_bps": cost_info["slippage_bps"],
            "slippage_hkd": cost_info["slippage_hkd"],
            "total_friction_hkd": cost_info["total_frictions_hkd"],
            "adv_participation_pct": cost_info["participation_rate"] * 100.0,
            "requires_twap": cost_info["requires_twap"],
        })
    return orders


# ============================================================================
# Task 1.1, 1.2, 1.3 & 2.3: Look-Ahead Free Strategy Backtest Engine
# ============================================================================


def analyze_portfolio_health(
    portfolio_items: list[dict[str, Any]],
    total_capital: float,
    rf_rate: float = 0.04,
    max_single_weight: float = 0.35,
    min_adv_hkd: float = 5_000_000.0,
    alloc_method: str = "hrp",
    num_trials: int = 5,
    cov_estimator: str = "ledoit_wolf",
    max_turnover: float = 0.40,
    broker_comm_rate: float = DEFAULT_BROKER_COMM_RATE,
    min_broker_comm_hkd: float = DEFAULT_MIN_BROKER_COMM_HKD,
    almgren_gamma: float = DEFAULT_ALMGREN_GAMMA,
) -> dict[str, Any] | None:
    """
    Comprehensive Institutional Portfolio Health, Denoised Optimization & Backtest Workbench.
    """
    if not portfolio_items:
        return None

    adj_price_dict: dict[str, pd.Series] = {}
    raw_price_dict: dict[str, pd.Series] = {}
    susp_dict: dict[str, pd.Series] = {}
    adv_map: dict[str, float] = {}
    vol_map: dict[str, float] = {}
    valid_items: list[dict[str, Any]] = []
    skipped_tickers: list[str] = []
    insufficient_history_tickers: list[str] = []

    for item in portfolio_items:
        raw_t = str(item.get('ticker', '')).strip().upper()
        if not raw_t:
            continue
        df, matched_t = safe_download_ohlcv(raw_t, period="1y", interval="1d", min_rows=30)
        if df is not None and matched_t is not None:
            susp_s = df['Is_Suspended'] if 'Is_Suspended' in df.columns else (df['Volume'] <= 0)
            active_days = int((~susp_s).sum())
            if len(df) < MIN_REQUIRED_DAILY_BARS or active_days < int(MIN_REQUIRED_DAILY_BARS * 0.75):
                insufficient_history_tickers.append(f"{matched_t} (有效交易日 {active_days} < {MIN_REQUIRED_DAILY_BARS})")
                continue

            adj_price_dict[matched_t] = df['Close']
            raw_col = df['Raw_Close'] if 'Raw_Close' in df.columns else df['Close']
            raw_price_dict[matched_t] = raw_col
            susp_dict[matched_t] = susp_s
            vol_s = df['Volume'] if 'Volume' in df.columns else pd.Series(0.0, index=df.index)
            adv_val = float((raw_col * vol_s).where(~susp_s, np.nan).rolling(20, min_periods=5).mean().dropna().iloc[-1])
            adv_map[matched_t] = adv_val
            vol_map[matched_t] = float(raw_col.pct_change().where(~susp_s, np.nan).rolling(20, min_periods=5).std().dropna().iloc[-1])

            lot_sz = int(item.get('lot_size') or HK_DEFAULT_LOT_SIZES.get(matched_t, 100))
            valid_items.append({
                "ticker": matched_t,
                "name": item.get("name", matched_t),
                "shares": int(item.get("shares", 0)),
                "lot_size": lot_sz,
                "short_eligible": is_hk_short_eligible(matched_t, adv_map[matched_t], item.get("short_eligible")),
                "entry_price": item.get("entry_price"),
                "thesis": item.get("thesis", ""),
                "invalidation_rule": item.get("invalidation_rule", ""),
            })
        else:
            skipped_tickers.append(raw_t)

    if not adj_price_dict:
        return None

    adj_data = pd.DataFrame(adj_price_dict).sort_index().ffill(limit=5).dropna(how="any")
    raw_data = pd.DataFrame(raw_price_dict).sort_index().ffill(limit=5).reindex(adj_data.index).ffill()
    susp_data = pd.DataFrame(susp_dict).reindex(adj_data.index).fillna(True)

    if adj_data.empty or len(adj_data) < MIN_REQUIRED_DAILY_BARS:
        return None

    latest_raw_prices = raw_data.iloc[-1]
    shares_map = {item['ticker']: item['shares'] for item in valid_items}

    holdings_value = {}
    total_market_val = 0.0
    for t in adj_data.columns:
        if t in shares_map:
            p = float(latest_raw_prices[t])
            val = p * shares_map[t]
            holdings_value[t] = val
            total_market_val += val

    masked_rets, _ = compute_active_daily_returns(adj_data, suspension_df=susp_data)
    if masked_rets.dropna(how="all").empty:
        return None

    if total_market_val > 0:
        weights = {t: holdings_value[t] / total_market_val for t in holdings_value.keys()}
    else:
        weights = {t: 1.0 / len(adj_data.columns) for t in adj_data.columns}

    position_val_df = pd.DataFrame(
        {t: adj_data[t] * shares_map.get(t, 0) for t in adj_data.columns}
    )
    hist_total_val = position_val_df.sum(axis=1)
    if (hist_total_val > 0).all():
        portfolio_daily_ret = hist_total_val.pct_change().dropna()
    else:
        portfolio_daily_ret = masked_rets.mean(axis=1).dropna()

    n_days = max(1, len(portfolio_daily_ret))
    cum_nav = (1.0 + portfolio_daily_ret).cumprod()
    ann_return = float(cum_nav.iloc[-1] ** (252.0 / n_days) - 1.0)
    ann_volatility = float(portfolio_daily_ret.std() * np.sqrt(252))
    sharpe_ratio = float((ann_return - rf_rate) / (ann_volatility + 1e-9))

    var_95 = float(np.percentile(portfolio_daily_ret, 5))
    tail_losses = portfolio_daily_ret[portfolio_daily_ret <= var_95]
    cvar_95 = float(tail_losses.mean()) if not tail_losses.empty else var_95

    if (cov_estimator or "ledoit_wolf").lower() == "rmt":
        denoised_cov, shrink_delta = compute_rmt_denoised_cov(masked_rets)
    else:
        denoised_cov, shrink_delta = compute_ledoit_wolf_cov(masked_rets)

    d_std = np.sqrt(np.maximum(np.diag(denoised_cov), 1e-8))
    denoised_corr_vals = np.clip(denoised_cov / (np.outer(d_std, d_std) + 1e-12), -1.0, 1.0)
    np.fill_diagonal(denoised_corr_vals, 1.0)
    corr_matrix = pd.DataFrame(denoised_corr_vals, index=adj_data.columns, columns=adj_data.columns)

    avg_corr = (
        float(corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean())
        if len(weights) > 1
        else 0.0
    )

    effective_capital = float(total_capital) if total_capital > 0 else total_market_val
    optimal_weights = optimize_portfolio_weights(
        masked_rets.iloc[:-1],
        max_single_weight=max_single_weight,
        method=alloc_method,
        cov_estimator=cov_estimator,
        prev_weights=weights if total_market_val > 0 else None,
        max_turnover=max_turnover,
    )
    execution_orders = generate_execution_orders(
        valid_items,
        effective_capital,
        optimal_weights,
        latest_raw_prices,
        adv_map=adv_map,
        vol_map=vol_map,
        broker_comm_rate=broker_comm_rate,
        min_broker_comm_hkd=min_broker_comm_hkd,
        almgren_gamma=almgren_gamma,
    )
    try:
        from hk_quant_system.engine.backtest import run_strategy_backtest
    except (ImportError, ValueError):
        from engine.backtest import run_strategy_backtest
    backtest_res = run_strategy_backtest(
        adj_data,
        lookback=60,
        rebalance_freq=20,
        max_single_weight=max_single_weight,
        rf_rate=rf_rate,
        alloc_method=alloc_method,
        num_trials=num_trials,
        cov_estimator=cov_estimator,
        max_turnover=max_turnover,
        suspension_df=susp_data,
        adv_map=adv_map,
        total_capital=effective_capital,
        broker_comm_rate=broker_comm_rate,
        min_broker_comm_hkd=min_broker_comm_hkd,
        almgren_gamma=almgren_gamma,
    )

    health_score = 100
    warnings_list = []

    if skipped_tickers:
        warnings_list.append(f"⚠️ 以下代號無法取得有效數據並已略過：{', '.join(skipped_tickers)}")
    if insufficient_history_tickers:
        warnings_list.append(
            f"🛡️ 數據充分性保護啟動：以下上市或有效交易日不足 {MIN_REQUIRED_DAILY_BARS} 根之標的已自動排除，防止共變異數矩陣奇異：{', '.join(insufficient_history_tickers)}"
        )

    total_suspended_days = int(susp_data.sum().sum())
    if total_suspended_days > 0:
        warnings_list.append(
            f"🛡️ 停牌遮罩已啟動：共識別並剔除 {total_suspended_days} 個停牌或無交易日零報酬樣本，消除低波動率失真。"
        )

    est_label = "RMT 隨機矩陣特徵值去噪" if (cov_estimator or "").lower() == "rmt" else "Ledoit-Wolf 解析壓縮共變異數"
    warnings_list.append(
        f"📐 矩陣去噪引擎：採用 {est_label} (最優壓縮強度 δ={shrink_delta:.2f}) 與單次換手率上限 {max_turnover*100:.0f}% 約束。"
    )

    max_weight = max(weights.values()) if weights else 0.0
    if max_weight > max_single_weight:
        health_score -= 25
        warnings_list.append(
            f"⚠️ 單一股票權重過高 ({max_weight*100:.1f}% > 上限 {max_single_weight*100:.0f}%)，資產過度集中。"
        )
    elif max_weight > 0.25:
        health_score -= 10
        warnings_list.append(f"🟡 最大單一持倉佔比達 {max_weight*100:.1f}%。")
    else:
        warnings_list.append("🟢 單一持倉權重分配合理。")

    if avg_corr > 0.70:
        health_score -= 30
        warnings_list.append(f"🔴 組合內資產平均相關性高達 {avg_corr:.2f}，遭遇極端行情時缺乏防禦力。")
    elif avg_corr > 0.50:
        health_score -= 15
        warnings_list.append(f"🟡 資產關聯度偏高 ({avg_corr:.2f})。")
    else:
        warnings_list.append(f"🟢 資產關聯度健康 ({avg_corr:.2f})。")

    if cvar_95 < -0.035:
        health_score -= 15
        warnings_list.append(f"⚠️ 組合單日極端下行風險 (1-Day CVaR 95%) 達 {cvar_95*100:.2f}%。")

    illiquid_names = [t for t, adv in adv_map.items() if adv < min_adv_hkd]
    if illiquid_names:
        health_score -= 15
        warnings_list.append(
            f"⚠️ 低流動性持倉警示：{', '.join(illiquid_names)} 的 20日均成交額低於 ${min_adv_hkd:,.0f} HKD，調倉時需注意買賣價差衝擊。"
        )

    if backtest_res and not backtest_res["dsr_significant"]:
        warnings_list.append(
            f"🔬 抗過擬合統計檢定 (Deflated Sharpe Ratio): 信心水準為 {backtest_res['dsr_prob']*100:.1f}% (p={backtest_res['dsr_pvalue']:.3f} > 0.05)，"
            f"建議優先採用 HRP 或結合 Ledoit-Wolf 去噪配置。"
        )
    elif backtest_res:
        warnings_list.append(
            f"🟢 抗過擬合統計檢定 (Deflated Sharpe Ratio): 信心水準達 {backtest_res['dsr_prob']*100:.1f}% (p={backtest_res['dsr_pvalue']:.3f} <= 0.05)，通過多重測試檢定。"
        )

    health_score = max(0, health_score)
    health_status = (
        "🟢 組合健康良好"
        if health_score >= 80
        else ("🟡 組合風險中等" if health_score >= 60 else "🔴 組合風險偏高")
    )

    res_dict = {
        "health_score": health_score,
        "health_status": health_status,
        "total_value": total_market_val,
        "effective_capital": effective_capital,
        "weights": weights,
        "optimal_weights": optimal_weights,
        "execution_orders": execution_orders,
        "ann_volatility": ann_volatility,
        "ann_return": ann_return,
        "sharpe_ratio": sharpe_ratio,
        "cvar_95": cvar_95,
        "avg_correlation": avg_corr,
        "corr_matrix": corr_matrix,
        "shrinkage_delta": shrink_delta,
        "adv_map": adv_map,
        "backtest": backtest_res,
        "warnings": warnings_list,
    }
    log_snapshot_to_db(res_dict)
    return res_dict


# ============================================================================
# Task 3: Historical Prediction Accuracy & Multi-Horizon Backtest Module
# ============================================================================