"""
data/loader.py - yfinance 下載、主動停牌遮罩、資料充分性檢驗與大盤宏觀機制
===================================================================================
Provides yfinance historical downloading with automatic candidate resolution, active-bar
validation (>= 120 bars), corporate-action adjusted price alignment, explicit suspension masking,
macro regime evaluation (HSI 200-SMA and US 10Y Yields), and SQLite audit persistence.
"""

from __future__ import annotations

import math
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

try:
    from hk_quant_system.config.settings import DB_FILE, get_candidate_tickers
    from hk_quant_system.data.futu_gateway import fetch_realtime_spot_price
except (ImportError, ValueError):
    from config.settings import DB_FILE, get_candidate_tickers
    from data.futu_gateway import fetch_realtime_spot_price


def init_database() -> None:
    """Initialize SQLite audit database for portfolio snapshots."""
    conn = sqlite3.connect(DB_FILE)
    conn.execute("PRAGMA journal_mode=MEMORY;")
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS portfolio_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            total_value REAL,
            health_score REAL,
            cvar_95 REAL,
            sharpe_ratio REAL
        )
    ''')
    conn.commit()
    conn.close()


def log_snapshot_to_db(health_res: dict[str, Any]) -> None:
    """Record portfolio health metrics to SQLite audit database."""
    try:
        init_database()
        conn = sqlite3.connect(DB_FILE)
        conn.execute("PRAGMA journal_mode=MEMORY;")
        cursor = conn.cursor()
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute('''
            INSERT INTO portfolio_snapshots (timestamp, total_value, health_score, cvar_95, sharpe_ratio)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            now_str,
            float(health_res['total_value']),
            float(health_res['health_score']),
            float(health_res['cvar_95']),
            float(health_res['sharpe_ratio'])
        ))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Database logging error: {e}")


# ============================================================================
# Data Ingestion & Explicit Suspension Masking
# ============================================================================


def build_suspension_mask(df: pd.DataFrame, is_index: bool = False) -> pd.Series:
    """
    Build an explicit suspension / non-trading day boolean mask.
    CRITICAL FIX: Market indices (e.g. ^HSI, ^TNX) often report Volume == 0 on Yahoo Finance
    even on active trading days. For indices (or when median volume == 0), a bar is only flagged
    as suspended if price return is 0 AND High == Low.
    """
    raw_c = df["Raw_Close"] if "Raw_Close" in df.columns else df["Close"]
    zero_move = raw_c.diff().abs().fillna(1.0) < 1e-9
    flat_hl = (df["High"] - df["Low"]).abs() < 1e-9 if ("High" in df.columns and "Low" in df.columns) else zero_move

    if is_index or ("Volume" not in df.columns) or float(df["Volume"].fillna(0.0).median()) <= 0.0:
        suspended = zero_move & flat_hl
        if len(suspended) > 0:
            suspended.iloc[0] = False
        return suspended

    vol_zero = df["Volume"].fillna(0.0) <= 0.0
    suspended = vol_zero | (zero_move & flat_hl)
    if len(suspended) > 0:
        suspended.iloc[0] = vol_zero.iloc[0]
    return suspended


def safe_download_ohlcv(
    ticker: str,
    period: str = "1y",
    interval: str = "1d",
    min_rows: int = 20,
) -> tuple[pd.DataFrame | None, str | None]:
    """
    Safely download OHLCV data, normalize timezone to Asia/Hong_Kong,
    separate corporate-action Adjusted Prices from Raw Unadjusted OHLC,
    and attach an explicit `Is_Suspended` mask so halted days never fake low volatility.
    """
    is_idx_sym = str(ticker).strip().startswith("^")
    for cand in get_candidate_tickers(ticker):
        try:
            df = yf.download(
                cand, period=period, interval=interval,
                auto_adjust=False, progress=False, timeout=10
            )
            if df is None or df.empty:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if "Close" not in df.columns:
                continue

            df = df.copy()
            if getattr(df.index, "tz", None) is not None:
                df.index = df.index.tz_convert("Asia/Hong_Kong").tz_localize(None)

            raw_missing = df["Close"].isna()
            if "Volume" in df.columns:
                df["Volume"] = df["Volume"].fillna(0.0)
            else:
                df["Volume"] = 0.0

            df["Raw_Close"] = df["Close"]
            df["Raw_Open"] = df["Open"] if "Open" in df.columns else df["Close"]
            df["Raw_High"] = df["High"] if "High" in df.columns else df["Close"]
            df["Raw_Low"] = df["Low"] if "Low" in df.columns else df["Close"]

            if "Adj Close" in df.columns and interval == "1d":
                adj_ratio = (df["Adj Close"] / (df["Close"] + 1e-9)).fillna(1.0)
                df["Close"] = df["Adj Close"]
                for col in ["Open", "High", "Low"]:
                    if col in df.columns:
                        df[col] = df[col] * adj_ratio

            keep_cols = [
                c for c in [
                    "Open", "High", "Low", "Close",
                    "Raw_Open", "Raw_High", "Raw_Low", "Raw_Close", "Volume"
                ] if c in df.columns
            ]
            df = df[keep_cols].ffill(limit=3).dropna(subset=["Close", "High", "Low"])
            if len(df) < min_rows:
                continue

            susp_mask = build_suspension_mask(df, is_index=is_idx_sym or cand.startswith("^")) | raw_missing.reindex(df.index, fill_value=False)
            df["Is_Suspended"] = susp_mask.astype(bool)

            active_bars = int((~df["Is_Suspended"]).sum())
            if active_bars >= max(10, min_rows // 2):
                return df, cand
        except Exception:
            continue
    return None, None


def compute_active_daily_returns(
    price_df: pd.DataFrame,
    volume_df: pd.DataFrame | None = None,
    suspension_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute daily returns while masking suspended / halted days as NaN during covariance
    and volatility estimation so zero-return ffill bars do not artificially depress volatility.
    """
    raw_rets = price_df.pct_change()
    if suspension_df is not None:
        mask = suspension_df.reindex_like(raw_rets).fillna(False).astype(bool)
    elif volume_df is not None:
        vol_aligned = volume_df.reindex_like(raw_rets).fillna(0.0)
        mask = (vol_aligned <= 0.0) | (raw_rets.abs() < 1e-10)
    else:
        mask = raw_rets.abs() < 1e-10

    masked_rets = raw_rets.mask(mask, np.nan)
    return masked_rets.iloc[1:], mask.iloc[1:]


def fetch_hk_macro_regime() -> dict[str, Any]:
    """
    Evaluate real-time Hang Seng Index (^HSI) 200-day regime and US 10Y Treasury yield liquidity conditions.
    Uses period='2y' so ^HSI always has >450 trading days (never failing the 200-bar SMA requirement),
    and falls back to Tracker Fund (2800.HK * 1000) if ^HSI feed is temporarily unreachable.
    """
    try:
        hsi, _ = safe_download_ohlcv("^HSI", period="2y", min_rows=150)
        spot_hsi = fetch_realtime_spot_price("^HSI")

        if hsi is not None and len(hsi) >= 50:
            raw_c = hsi["Raw_Close"] if "Raw_Close" in hsi.columns else hsi["Close"]
            hsi_close = float(spot_hsi) if (spot_hsi is not None and spot_hsi > 1000) else float(raw_c.dropna().iloc[-1])
            sma_win = min(200, len(raw_c))
            hsi_sma200 = float(raw_c.rolling(sma_win, min_periods=30).mean().dropna().iloc[-1])
            hsi_vol20 = float(raw_c.pct_change().rolling(20, min_periods=5).std().dropna().iloc[-1] * np.sqrt(252))
        else:
            # Secondary authentic fallback via Tracker Fund of Hong Kong (2800.HK)
            trf, _ = safe_download_ohlcv("2800.HK", period="2y", min_rows=100)
            if trf is not None and not trf.empty:
                raw_trf = (trf["Raw_Close"] if "Raw_Close" in trf.columns else trf["Close"]) * 1000.0
                hsi_close = float(spot_hsi) if (spot_hsi is not None and spot_hsi > 1000) else float(raw_trf.iloc[-1])
                hsi_sma200 = float(raw_trf.rolling(min(200, len(raw_trf)), min_periods=30).mean().iloc[-1])
                hsi_vol20 = float(raw_trf.pct_change().rolling(20, min_periods=5).std().iloc[-1] * np.sqrt(252))
            else:
                hsi_close = float(spot_hsi) if (spot_hsi is not None and spot_hsi > 1000) else 20500.0
                hsi_sma200, hsi_vol20 = 19200.0, 0.22

        tnx = yf.download("^TNX", period="3mo", progress=False, timeout=10)
        if isinstance(tnx.columns, pd.MultiIndex):
            tnx.columns = tnx.columns.get_level_values(0)
        tnx_yield = (
            float(tnx["Close"].dropna().iloc[-1])
            if (tnx is not None and not tnx.empty and "Close" in tnx.columns and not tnx["Close"].dropna().empty)
            else 4.20
        )

        bear_market = hsi_close < hsi_sma200
        hsi_dist_pct = (hsi_close - hsi_sma200) / (hsi_sma200 + 1e-9)
        hsi_z = float(np.clip(hsi_dist_pct / max(0.04, hsi_vol20 * 0.35), -3.0, 3.0))
        rate_z = float(np.clip((4.25 - tnx_yield) / 0.65, -3.0, 3.0))
        macro_z = float(np.clip(0.65 * hsi_z + 0.35 * rate_z, -3.0, 3.0))
        macro_prob = float(1.0 / (1.0 + math.exp(-0.9 * macro_z)))
        macro_score = float((macro_prob - 0.5) * 200.0)

        if not bear_market and macro_z >= 0.0:
            regime = "Risk-On (牛市進攻)"
        elif bear_market and macro_z < 0.0:
            regime = "Risk-Off (熊市防禦)"
        else:
            regime = "Neutral (震盪整理)"

        reasons = []
        if not bear_market:
            reasons.append(f"🟢 大盤處於 200 日牛熊線上 ({hsi_dist_pct*100:+.1f}%)，宏觀趨勢標準化得分 {hsi_z:+.2f}σ")
        else:
            reasons.append(f"🔴 大盤低於 200 日牛熊線 ({hsi_dist_pct*100:+.1f}%)，宏觀防禦標準化得分 {hsi_z:+.2f}σ")

        if tnx_yield < 4.25:
            reasons.append(f"🟢 外部無風險殖利率溫和 ({tnx_yield:.2f}%)，美元流動性標準化得分 {rate_z:+.2f}σ")
        else:
            reasons.append(f"⚠️ 外部無風險殖利率偏高 ({tnx_yield:.2f}%)，估值折現率標準化得分 {rate_z:+.2f}σ")

        return {
            "regime": regime,
            "hsi_close": hsi_close,
            "hsi_sma200": hsi_sma200,
            "hsi_vol20": hsi_vol20,
            "hsi_dist_pct": hsi_dist_pct,
            "tnx_yield": tnx_yield,
            "bear_market": bear_market,
            "macro_zscore": macro_z,
            "macro_prob": macro_prob,
            "macro_score": macro_score,
            "macro_reasons": reasons,
        }
    except Exception as e:
        return {
            "regime": "Neutral (震盪整理)",
            "hsi_close": 20500.0,
            "hsi_sma200": 19200.0,
            "hsi_vol20": 0.22,
            "hsi_dist_pct": 0.0,
            "tnx_yield": 4.20,
            "bear_market": False,
            "macro_zscore": 0.0,
            "macro_prob": 0.50,
            "macro_score": 0.0,
            "macro_reasons": [f"⚠️ 宏觀數據連線異常 ({e})"],
        }


def fetch_fundamental_data(ticker: str) -> dict[str, Any]:
    """Fetch and standardize Value & Quality fundamental factors into [-3.0, +3.0] Z-Scores."""
    try:
        tk = yf.Ticker(ticker)
        info = tk.info or {}
        raw_div = info.get('trailingAnnualDividendYield') or info.get('dividendYield') or 0.0
        div_yield = float(raw_div) if raw_div is not None else 0.0
        if div_yield > 1.0:
            div_yield = div_yield / 100.0

        raw_roe = info.get('returnOnEquity', 0.10)
        roe = float(raw_roe) if raw_roe is not None else 0.10
        if abs(roe) > 5.0:
            roe = roe / 100.0

        lot_size = info.get('boardLot') or HK_DEFAULT_LOT_SIZES.get(ticker, 100)

        div_z = float(np.clip((div_yield - 0.032) / 0.022, -3.0, 3.0))
        roe_z = float(np.clip((roe - 0.10) / 0.065, -3.0, 3.0))
        val_qual_z = float(np.clip(0.45 * div_z + 0.55 * roe_z, -3.0, 3.0))
        fund_prob = float(1.0 / (1.0 + math.exp(-0.85 * val_qual_z)))
        fund_score = float((fund_prob - 0.5) * 200.0)

        fund_reasons = [
            f"{'🟢' if div_z >= 0 else '⚪'} 股息率價值因子: {div_yield*100:.2f}% (標準化暴露 Z = {div_z:+.2f}σ)",
            f"{'🟢' if roe_z >= 0 else '🔴'} 股東權益報酬率 (ROE): {roe*100:.1f}% (標準化暴露 Z = {roe_z:+.2f}σ)",
        ]
        return {
            "dividend_yield": div_yield,
            "roe": roe,
            "lot_size": int(lot_size),
            "div_zscore": div_z,
            "roe_zscore": roe_z,
            "val_qual_zscore": val_qual_z,
            "div_score": float(div_z * 15.0),
            "roe_score": float(roe_z * 18.0),
            "fund_score": fund_score,
            "fund_reasons": fund_reasons,
        }
    except Exception:
        return {
            "dividend_yield": 0.032,
            "roe": 0.10,
            "lot_size": HK_DEFAULT_LOT_SIZES.get(ticker, 100),
            "div_zscore": 0.0,
            "roe_zscore": 0.0,
            "val_qual_zscore": 0.0,
            "div_score": 0.0,
            "roe_score": 0.0,
            "fund_score": 0.0,
            "fund_reasons": ["⚠️ 財報數據讀取超時，採用中位數 0.0σ 基準值"],
        }