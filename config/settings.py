"""
config/settings.py - 港交所規費、合規名單、全域常數與路徑設定
===================================================================================
Provides statutory HKEX spreads, tick sizes, fees, short-sale eligibility whitelist,
default board lot sizes, and atomic configuration persistence.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

BASE_DIR: Path = Path(__file__).resolve().parent.parent
PORTFOLIO_FILE: str = str(BASE_DIR / "portfolio.json")
DB_FILE: str = str(BASE_DIR / "quant_system_audit.db")
MIN_REQUIRED_DAILY_BARS: int = 120

HK_DEFAULT_LOT_SIZES: dict[str, int] = {
    "0700.HK": 100,
    "0005.HK": 400,
    "3690.HK": 100,
    "9988.HK": 100,
    "2475.HK": 100,
    "1299.HK": 200,
    "0388.HK": 100,
    "0941.HK": 500,
    "0001.HK": 500,
    "2318.HK": 500,
    "1398.HK": 1000,
    "3988.HK": 1000,
    "0883.HK": 1000,
    "1810.HK": 200,
    "9618.HK": 50,
    "9999.HK": 100,
    "1211.HK": 500,
}

# Major SFC Designated Securities Eligible for Short Selling (liquid large-caps)
HK_SHORT_ELIGIBLE_WHITELIST: set[str] = {
    "0700.HK", "0005.HK", "3690.HK", "9988.HK", "1299.HK",
    "0388.HK", "0941.HK", "0001.HK", "2318.HK", "1398.HK",
    "3988.HK", "0883.HK", "1810.HK", "9618.HK", "9999.HK",
    "1211.HK", "0016.HK", "0002.HK", "0003.HK", "0011.HK",
    "0027.HK", "0066.HK", "0175.HK", "0267.HK", "0288.HK",
    "0386.HK", "0669.HK", "0688.HK", "0762.HK", "0823.HK",
    "0857.HK", "0939.HK", "0968.HK", "0981.HK", "1024.HK",
    "1088.HK", "1093.HK", "1109.HK", "1177.HK", "1928.HK",
    "2015.HK", "2020.HK", "2269.HK", "2313.HK", "2319.HK",
    "2331.HK", "2382.HK", "2388.HK", "2628.HK", "3692.HK",
    "6690.HK", "9633.HK", "9868.HK", "9961.HK",
}

# Statutory HKEX Fee Rates
HKEX_STAMP_DUTY_RATE: float = 0.0010        # 0.10%, rounded UP to nearest HK$1.00
HKEX_SFC_LEVY_RATE: float = 0.000027        # 0.0027%
HKEX_AFRC_LEVY_RATE: float = 0.0000015      # 0.00015%
HKEX_TRADING_FEE_RATE: float = 0.0000565    # 0.00565%
DEFAULT_BROKER_COMM_RATE: float = 0.0003    # 0.03% default commission
DEFAULT_MIN_BROKER_COMM_HKD: float = 15.0   # HK$15 minimum commission
DEFAULT_ALMGREN_GAMMA: float = 0.35         # Almgren-Chriss market impact coefficient


def get_hk_tick_size(price: float) -> float:
    """Return statutory HKEX Main Board tick size (spread) for a given price in HKD."""
    if price < 0.25:
        return 0.001
    if price < 0.50:
        return 0.005
    if price < 10.00:
        return 0.01
    if price < 20.00:
        return 0.02
    if price < 100.00:
        return 0.05
    if price < 200.00:
        return 0.10
    if price < 500.00:
        return 0.20
    if price < 1000.00:
        return 0.50
    if price < 2000.00:
        return 1.00
    if price < 5000.00:
        return 2.00
    return 5.00


def round_to_hk_tick(price: float) -> float:
    """Round any target, execution, or stop-loss price to the nearest valid HKEX statutory tick size."""
    if price <= 0 or np.isnan(price):
        return 0.0
    tick = get_hk_tick_size(price)
    rounded = round(round(price / tick) * tick, 3)
    return float(rounded)


def normalize_hk_ticker(ticker: str) -> str:
    """Normalize HK ticker to Yahoo Finance standard (4-digit zero-padded or 5-digit .HK)."""
    if not ticker:
        return ""
    raw = str(ticker).strip().upper()
    if raw.startswith("^"):
        return raw
    parts = raw.split(".")
    digits = parts[0].lstrip("0")
    if not digits.isdigit():
        return raw if raw.endswith(".HK") else f"{raw}.HK"
    num = int(digits)
    return f"{num:04d}.HK" if num < 10000 else f"{num:05d}.HK"


def get_candidate_tickers(ticker: str) -> list[str]:
    """Generate prioritized candidate tickers for yfinance lookup."""
    norm = normalize_hk_ticker(ticker)
    raw = str(ticker).strip().upper()
    digits = raw.split(".")[0].lstrip("0")
    if not digits.isdigit():
        return [raw]
    candidates = [norm, f"{digits.zfill(4)}.HK", f"{digits.zfill(5)}.HK", raw]
    return list(dict.fromkeys(candidates))


def is_hk_short_eligible(ticker: str, adv_20d_hkd: float = 0.0, config_flag: bool | None = None) -> bool:
    """
    Check if a HK stock is eligible for short selling under SFC Designated List rules.
    Small-cap or low-ADV stocks cannot be legally short-sold or lack stock borrow inventory.
    """
    if config_flag is not None:
        return bool(config_flag)
    norm_t = normalize_hk_ticker(ticker)
    if norm_t in HK_SHORT_ELIGIBLE_WHITELIST:
        return True
    return float(adv_20d_hkd) >= 30_000_000.0


# ============================================================================
# HKEX Microstructure Fee & Almgren-Chriss Market Impact Slippage Engine
# ============================================================================
def calculate_hkex_microstructure_cost(
    trade_value_hkd: float,
    adv_20d_hkd: float = 50_000_000.0,
    daily_volatility: float = 0.02,
    broker_comm_rate: float = DEFAULT_BROKER_COMM_RATE,
    min_broker_comm_hkd: float = DEFAULT_MIN_BROKER_COMM_HKD,
    gamma: float = DEFAULT_ALMGREN_GAMMA,
    include_stamp_duty: bool = True,
) -> dict[str, Any]:
    """
    Compute exact HKEX statutory transaction fees and Almgren-Chriss square-root market impact slippage.

    Statutory Formulae:
    - Hong Kong Stamp Duty: 0.10%, rounded UP to the nearest HK$1.00 (`math.ceil(trade_val * 0.001)`)
    - SFC Transaction Levy: 0.0027%
    - AFRC Transaction Levy: 0.00015%
    - HKEX Trading Fee: 0.00565%
    - Brokerage Commission: max(trade_val * broker_comm_rate, min_broker_comm_hkd)
    - Almgren-Chriss Square-Root Market Impact:
      Slippage (bps) = gamma * sqrt(trade_val / ADV_20) * sigma_daily * 10,000
      With an additional liquidity penalty multiplier when single-order participation > 5% of ADV_20.
    """
    trade_val = max(0.0, float(trade_value_hkd))
    if trade_val <= 0.0:
        return {
            "trade_value_hkd": 0.0,
            "stamp_duty_hkd": 0.0,
            "sfc_levy_hkd": 0.0,
            "afrc_levy_hkd": 0.0,
            "hkex_fee_hkd": 0.0,
            "broker_commission_hkd": 0.0,
            "total_statutory_fees_hkd": 0.0,
            "statutory_fee_bps": 0.0,
            "participation_rate": 0.0,
            "slippage_bps": 0.0,
            "slippage_hkd": 0.0,
            "total_frictions_hkd": 0.0,
            "total_cost_bps": 0.0,
            "requires_twap": False,
            "recommended_twap_slices": 1,
        }

    stamp_duty = float(math.ceil(trade_val * HKEX_STAMP_DUTY_RATE)) if include_stamp_duty else 0.0
    sfc_levy = round(trade_val * HKEX_SFC_LEVY_RATE, 2)
    afrc_levy = round(trade_val * HKEX_AFRC_LEVY_RATE, 2)
    hkex_fee = round(trade_val * HKEX_TRADING_FEE_RATE, 2)
    broker_comm = max(round(trade_val * max(0.0, float(broker_comm_rate)), 2), float(min_broker_comm_hkd))

    total_statutory = stamp_duty + sfc_levy + afrc_levy + hkex_fee + broker_comm
    statutory_bps = (total_statutory / trade_val) * 10000.0

    eff_adv = max(float(adv_20d_hkd), 50_000.0)
    sigma_d = float(np.clip(float(daily_volatility), 0.003, 0.25))
    participation = trade_val / eff_adv

    # Almgren-Chriss square-root temporary + permanent impact model
    base_slippage_bps = float(gamma) * math.sqrt(participation) * (sigma_d * 10000.0)

    # Nonlinear liquidity penalty when participation exceeds 5% of 20-day ADV
    requires_twap = participation > 0.05
    if requires_twap:
        excess_ratio = participation / 0.05
        liquidity_penalty_mult = 1.0 + 0.35 * math.log1p(excess_ratio - 1.0)
        slippage_bps = min(base_slippage_bps * liquidity_penalty_mult, 500.0)
        recommended_slices = max(2, int(math.ceil(participation / 0.025)))
    else:
        slippage_bps = min(base_slippage_bps, 300.0)
        recommended_slices = 1

    slippage_hkd = trade_val * (slippage_bps / 10000.0)
    total_frictions_hkd = total_statutory + slippage_hkd
    total_cost_bps = statutory_bps + slippage_bps

    return {
        "trade_value_hkd": trade_val,
        "stamp_duty_hkd": stamp_duty,
        "sfc_levy_hkd": sfc_levy,
        "afrc_levy_hkd": afrc_levy,
        "hkex_fee_hkd": hkex_fee,
        "broker_commission_hkd": broker_comm,
        "total_statutory_fees_hkd": total_statutory,
        "statutory_fee_bps": statutory_bps,
        "participation_rate": participation,
        "slippage_bps": slippage_bps,
        "slippage_hkd": slippage_hkd,
        "total_frictions_hkd": total_frictions_hkd,
        "total_cost_bps": total_cost_bps,
        "requires_twap": requires_twap,
        "recommended_twap_slices": recommended_slices,
    }


# ============================================================================
# Atomic Configuration Persistence & Database Audit
# ============================================================================
def load_portfolio_config(filepath: str = PORTFOLIO_FILE) -> dict[str, Any]:
    """Load portfolio configuration from portfolio.json with safe defaults and schema normalization."""
    default_config: dict[str, Any] = {
        "total_capital_hkd": 300000.0,
        "risk_free_rate": 0.04,
        "max_single_position_pct": 0.35,
        "min_adv_hkd": 5000000.0,
        "alloc_method": "hrp",
        "cov_estimator": "ledoit_wolf",
        "max_turnover": 0.40,
        "broker_comm_rate": DEFAULT_BROKER_COMM_RATE,
        "min_broker_comm_hkd": DEFAULT_MIN_BROKER_COMM_HKD,
        "almgren_gamma": DEFAULT_ALMGREN_GAMMA,
        "num_backtest_trials": 5,
        "holdings": [
            {
                "ticker": "0700.HK",
                "name": "騰訊控股",
                "lot_size": 100,
                "shares": 300,
                "entry_price": 360.0,
                "short_eligible": True,
                "roic": 0.16,
                "debt_to_ebitda": 1.2,
                "thesis": "遊戲版號常態化，視頻號廣告毛利率持續擴張",
                "invalidation_rule": "核心遊戲收入連續兩季按年下跌超過 5%",
            },
            {
                "ticker": "0005.HK",
                "name": "匯豐控股",
                "lot_size": 400,
                "shares": 1200,
                "entry_price": 68.0,
                "short_eligible": True,
                "roic": 0.11,
                "debt_to_ebitda": 2.5,
                "thesis": "高息環境維持淨息差，持續進行大規模股份回購",
                "invalidation_rule": "宣布暫停派息或停止回購計畫",
            },
            {
                "ticker": "3690.HK",
                "name": "美團-W",
                "lot_size": 100,
                "shares": 400,
                "entry_price": 130.0,
                "short_eligible": True,
                "roic": 0.08,
                "debt_to_ebitda": 1.8,
                "thesis": "到店業務競爭趨緩，外賣單均盈利能力穩固",
                "invalidation_rule": "抖音本地生活市佔率進一步蠶食導致營業利潤轉負",
            },
            {
                "ticker": "2475.HK",
                "name": "立訊精密",
                "lot_size": 100,
                "shares": 100,
                "entry_price": 44.0,
                "short_eligible": False,
                "roic": 0.12,
                "debt_to_ebitda": 1.5,
                "thesis": "業務穩健增長",
                "invalidation_rule": "核心業務數據顯著惡化",
            },
        ],
    }
    if not os.path.exists(filepath):
        return default_config
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return default_config
        for h in data.get("holdings", []):
            h["ticker"] = normalize_hk_ticker(h.get("ticker", ""))
            h["lot_size"] = int(h.get("lot_size") or HK_DEFAULT_LOT_SIZES.get(h["ticker"], 100))
            h["shares"] = int(h.get("shares") or 0)
            if "short_eligible" not in h:
                h["short_eligible"] = h["ticker"] in HK_SHORT_ELIGIBLE_WHITELIST
        default_config.update(data)
        return default_config
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return default_config


def save_portfolio_config(config_data: dict[str, Any], filepath: str = PORTFOLIO_FILE) -> bool:
    """
    Atomically persist updated portfolio holdings and parameters back to portfolio.json.
    Prevents race conditions and partial file corruption during reactive Streamlit reruns.
    """
    try:
        target_dir = os.path.dirname(os.path.abspath(filepath)) or "."
        os.makedirs(target_dir, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(prefix=".portfolio_tmp_", suffix=".json", dir=target_dir)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as tmp_file:
                json.dump(config_data, tmp_file, ensure_ascii=False, indent=4)
                tmp_file.flush()
                os.fsync(tmp_file.fileno())
            os.replace(tmp_path, filepath)
            return True
        except Exception:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise
    except Exception as e:
        print(f"Error saving {filepath}: {e}")
        return False