"""
web/app.py - Streamlit 主介面與頁籤導航控制
===================================================================================
Main institutional quantitative trading terminal preserving the full 4-tab layout,
reactive sidebar, and real-time multi-factor portal showcase.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

_CURRENT_DIR = Path(__file__).resolve().parent
_PACKAGE_ROOT = _CURRENT_DIR.parent
_REPO_ROOT = _PACKAGE_ROOT.parent

for _p in [str(_PACKAGE_ROOT), str(_REPO_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from hk_quant_system.config.settings import (
        DEFAULT_ALMGREN_GAMMA,
        DEFAULT_BROKER_COMM_RATE,
        DEFAULT_MIN_BROKER_COMM_HKD,
        HK_DEFAULT_LOT_SIZES,
        HK_SHORT_ELIGIBLE_WHITELIST,
        is_hk_short_eligible,
        load_portfolio_config,
        normalize_hk_ticker,
        save_portfolio_config,
    )
    from hk_quant_system.data.futu_gateway import (
        check_futu_opend_connection,
        fetch_realtime_spot_price,
        get_futu_order_book,
        get_futu_realtime_quote,
        to_futu_ticker,
    )
    from hk_quant_system.data.loader import fetch_hk_macro_regime
    from hk_quant_system.engine.alpha import (
        analyze_stock_trend,
        generate_day_trading_signal,
    )
    from hk_quant_system.engine.risk import analyze_portfolio_health
    from hk_quant_system.engine.backtest import evaluate_historical_prediction_accuracy
    from hk_quant_system.web.components import (
        inject_terminal_and_portal_css,
        render_factor_bars_html,
    )
except (ImportError, ValueError):
    from config.settings import (
        DEFAULT_ALMGREN_GAMMA,
        DEFAULT_BROKER_COMM_RATE,
        DEFAULT_MIN_BROKER_COMM_HKD,
        HK_DEFAULT_LOT_SIZES,
        HK_SHORT_ELIGIBLE_WHITELIST,
        is_hk_short_eligible,
        load_portfolio_config,
        normalize_hk_ticker,
        save_portfolio_config,
    )
    from data.futu_gateway import (
        check_futu_opend_connection,
        fetch_realtime_spot_price,
        get_futu_order_book,
        get_futu_realtime_quote,
        to_futu_ticker,
    )
    from data.loader import fetch_hk_macro_regime
    from engine.alpha import (
        analyze_stock_trend,
        generate_day_trading_signal,
    )
    from engine.risk import analyze_portfolio_health
    from engine.backtest import evaluate_historical_prediction_accuracy
    from web.components import (
        inject_terminal_and_portal_css,
        render_factor_bars_html,
    )

st.set_page_config(
    page_title="香港股市預測 | HKEX AlphaEngine 機構級量化終端",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_terminal_and_portal_css()

# ==========================================
# Cached Quantitative Data Pipelines
# ==========================================
@st.cache_data(ttl=5, show_spinner=False)
def cached_order_book(ticker: str, host: str, port: int, spot_price: float, lot_size: int):
    return get_futu_order_book(
        ticker,
        num=10,
        host=host,
        port=port,
        spot_price_fallback=spot_price,
        lot_size_fallback=lot_size,
    )


@st.cache_data(ttl=60, show_spinner=False)
def cached_macro_regime() -> dict:
    return fetch_hk_macro_regime()


@st.cache_data(ttl=60, show_spinner=False)
def cached_day_trading_signal(
    ticker: str,
    min_adv_hkd: float,
    interval: str,
    broker_comm_rate: float,
    min_broker_comm_hkd: float,
    almgren_gamma: float,
):
    return generate_day_trading_signal(
        ticker,
        min_adv_hkd=min_adv_hkd,
        interval=interval,
        broker_comm_rate=broker_comm_rate,
        min_broker_comm_hkd=min_broker_comm_hkd,
        almgren_gamma=almgren_gamma,
    )


@st.cache_data(ttl=60, show_spinner=False)
def cached_stock_trend(ticker: str):
    return analyze_stock_trend(ticker)


@st.cache_data(ttl=300, show_spinner=False)
def cached_portfolio_health(
    holdings_json: str,
    total_cap: float,
    rf: float,
    max_w: float,
    min_adv: float,
    method: str,
    trials: int,
    cov_est: str,
    max_to: float,
    comm_rate: float,
    min_comm: float,
    gamma: float,
):
    holdings = json.loads(holdings_json)
    return analyze_portfolio_health(
        holdings,
        total_capital=total_cap,
        rf_rate=rf,
        max_single_weight=max_w,
        min_adv_hkd=min_adv,
        alloc_method=method,
        num_trials=trials,
        cov_estimator=cov_est,
        max_turnover=max_to,
        broker_comm_rate=comm_rate,
        min_broker_comm_hkd=min_comm,
        almgren_gamma=gamma,
    )


@st.cache_data(ttl=300, show_spinner=False)
def cached_historical_accuracy(
    ticker: str,
    horizon: int,
    history_period: str,
    atr_stop_mult: float,
    signal_z_threshold: float,
    comm_rate: float,
    min_comm: float,
    gamma: float,
):
    return evaluate_historical_prediction_accuracy(
        ticker=ticker,
        forward_horizon=horizon,
        history_period=history_period,
        atr_stop_mult=atr_stop_mult,
        signal_z_threshold=signal_z_threshold,
        broker_comm_rate=comm_rate,
        min_broker_comm_hkd=min_comm,
        almgren_gamma=gamma,
    )

@st.cache_data(ttl=60, show_spinner=False)
def _compute_single_portal_card(code: str, name: str, f_host: str, f_port: int) -> dict[str, Any]:
    """Compute real-time spot price and authentic multi-factor trend probability for a single showcase stock."""
    norm_c = normalize_hk_ticker(code)
    spot_p = fetch_realtime_spot_price(norm_c, futu_host=f_host, futu_port=f_port)
    trend_res, _ = analyze_stock_trend(norm_c)
    if trend_res:
        price_val = float(spot_p) if (spot_p is not None and spot_p > 0) else float(trend_res.get("price", 0.0))
        prob_pct = float(trend_res.get("up_prob", 0.5) * 100.0)
        z_val = float(trend_res.get("composite_zscore", 0.0))
        if prob_pct >= 58.0:
            tag, color = "看多趨勢", "#34D399"
        elif prob_pct <= 42.0:
            tag, color = "弱勢防禦", "#FB7185"
        elif prob_pct >= 52.0:
            tag, color = "偏多震盪", "#34D399"
        else:
            tag, color = "區間震盪", "#38BDF8"
        return {
            "code": norm_c,
            "name": name,
            "price": price_val,
            "prob": prob_pct,
            "z": z_val,
            "tag": tag,
            "color": color,
        }
    return {
        "code": norm_c,
        "name": name,
        "price": float(spot_p) if (spot_p is not None and spot_p > 0) else 0.0,
        "prob": 50.0,
        "z": 0.0,
        "tag": "即時連線中",
        "color": "#38BDF8",
    }


def get_live_portal_showcase_cards(cfg_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Fetch 4 showcase stocks in parallel using ThreadPoolExecutor with real market prices & factor model."""
    f_host = str(cfg_data.get("futu_host", "127.0.0.1"))
    f_port = int(cfg_data.get("futu_port", 11111))
    default_Names = {
        "0700.HK": "騰訊控股",
        "0005.HK": "匯豐控股",
        "3690.HK": "美團-W",
        "0388.HK": "香港交易所",
        "9988.HK": "阿里巴巴-W",
        "1810.HK": "小米集團-W",
        "1211.HK": "比亞迪股份",
        "2475.HK": "立訊精密",
    }
    target_pairs: list[tuple[str, str]] = []
    for h in cfg_data.get("holdings", []):
        tk = normalize_hk_ticker(h.get("ticker", ""))
        if tk and tk not in [p[0] for p in target_pairs]:
            target_pairs.append((tk, str(h.get("name", default_Names.get(tk, tk)))))
        if len(target_pairs) >= 4:
            break
    for fallback_tk in ["0700.HK", "0005.HK", "3690.HK", "0388.HK"]:
        if len(target_pairs) >= 4:
            break
        if fallback_tk not in [p[0] for p in target_pairs]:
            target_pairs.append((fallback_tk, default_Names[fallback_tk]))

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [
            pool.submit(_compute_single_portal_card, tk, nm, f_host, f_port)
            for tk, nm in target_pairs[:4]
        ]
        return [f.result() for f in futures]


# ==========================================
# Task 1.5: Reactive State & Atomic JSON Sync
# ==========================================
if "portfolio_config" not in st.session_state:
    st.session_state.portfolio_config = load_portfolio_config()
if "last_saved_hash" not in st.session_state:
    st.session_state.last_saved_hash = json.dumps(st.session_state.portfolio_config, sort_keys=True, ensure_ascii=False)
if "entered_main_terminal" not in st.session_state:
    # Show the fancy "香港股市預測" special-effects portal page first before the main page
    st.session_state.entered_main_terminal = False

cfg = st.session_state.portfolio_config

QUICK_WATCHLIST = {
    "0700.HK": "0700.HK 騰訊控股",
    "0005.HK": "0005.HK 匯豐控股",
    "3690.HK": "3690.HK 美團-W",
    "9988.HK": "9988.HK 阿里巴巴-W",
    "1810.HK": "1810.HK 小米集團-W",
    "0388.HK": "0388.HK 香港交易所",
    "1211.HK": "1211.HK 比亞迪股份",
    "2475.HK": "2475.HK 立訊精密/中小型股",
}

with st.sidebar:
    st.markdown("### 🎛️ 香港股市預測控制中心")
    if st.session_state.entered_main_terminal:
        if st.button("✨ 返回「香港股市預測」特效前導頁", use_container_width=True):
            st.session_state.entered_main_terminal = False
            st.rerun()
    else:
        if st.button("🚀 直接進入量化分析主終端", use_container_width=True, type="primary"):
            st.session_state.entered_main_terminal = True
            st.rerun()

    if st.button("🔄 立即強制重整最新即時報價", use_container_width=True):
        st.cache_data.clear()
        st.toast("⚡ 已強制清除所有快取，正在向市場重新抓取最新即時報價！")
        st.rerun()

    with st.expander("🔌 富途牛牛 (Futu Open API) L2 閘道配置", expanded=bool(cfg.get("enable_futu", False))):
        futu_enabled = st.checkbox("啟用富途 Level 2 串流報價", value=bool(cfg.get("enable_futu", False)), key="futu_enable_toggle")
        f_host = st.text_input("FutuOpenD 監聽位址", value=str(cfg.get("futu_host", "127.0.0.1")), key="futu_host_input")
        f_port = st.number_input("連接埠 (Port)", min_value=1000, max_value=65535, value=int(cfg.get("futu_port", 11111)), step=1, key="futu_port_input")
        
        cfg["enable_futu"] = futu_enabled
        cfg["futu_host"] = f_host
        cfg["futu_port"] = int(f_port)

        test_col1, test_col2 = st.columns([1, 1])
        with test_col1:
            if st.button("測試閘道連線", use_container_width=True):
                is_ok = check_futu_opend_connection(f_host, int(f_port), timeout=0.6)
                if is_ok:
                    st.success("🟢 FutuOpenD 已連線！(L2 即時串流生效)")
                else:
                    st.info("⚪ FutuOpenD 未啟動 (使用 Yahoo 備援)")
        with test_col2:
            is_connected_live = check_futu_opend_connection(f_host, int(f_port), timeout=0.2)
            status_text = "🟢 L2 串流中" if is_connected_live else "⚪ Yahoo 備援"
            st.metric("串流狀態", status_text)

    st.markdown("#### 🎯 單股追蹤標的")
    preset_t = st.selectbox(
        "快速切換熱門港股",
        options=list(QUICK_WATCHLIST.keys()),
        format_func=lambda k: QUICK_WATCHLIST[k],
        index=0,
    )
    custom_t = st.text_input("或輸入任意港股代號 (如 0941.HK)", value=preset_t).strip().upper()
    active_ticker = normalize_hk_ticker(custom_t) if custom_t else preset_t

    interval_choice = st.segmented_control(
        "日內當沖 K 線週期",
        options=["15m", "5m", "1d"],
        default="15m",
    ) or "15m"

    st.markdown("#### 🛡️ 投資組合與矩陣去噪設定")
    method_options = {
        "hrp": "HRP 階層風險平價 (推薦)",
        "risk_parity": "Risk Parity 等風險貢獻",
        "min_variance": "Minimum Variance 最小變異數",
    }
    curr_method = cfg.get("alloc_method", "hrp")
    alloc_method = st.selectbox(
        "資產配置數學模型",
        options=list(method_options.keys()),
        format_func=lambda k: method_options[k],
        index=list(method_options.keys()).index(curr_method) if curr_method in method_options else 0,
    )

    cov_options = {
        "ledoit_wolf": "Ledoit-Wolf 壓縮共變異數",
        "rmt": "RMT 隨機矩陣特徵值去噪",
    }
    curr_cov = cfg.get("cov_estimator", "ledoit_wolf")
    cov_estimator = st.selectbox(
        "共變異數矩陣去噪引擎",
        options=list(cov_options.keys()),
        format_func=lambda k: cov_options[k],
        index=list(cov_options.keys()).index(curr_cov) if curr_cov in cov_options else 0,
    )

    total_capital_hkd = st.number_input(
        "目標配置總資金 (HKD)",
        min_value=10000.0,
        value=float(cfg.get("total_capital_hkd", 300000.0)),
        step=10000.0,
    )
    rf_rate_pct = st.slider(
        "無風險利率 Rf (% 年化)",
        min_value=0.0,
        max_value=8.0,
        value=float(cfg.get("risk_free_rate", 0.04)) * 100.0,
        step=0.25,
    )
    max_pos_pct = st.slider(
        "單一持股權重上限 (%)",
        min_value=15.0,
        max_value=60.0,
        value=float(cfg.get("max_single_position_pct", 0.35)) * 100.0,
        step=5.0,
    )
    max_turnover_pct = st.slider(
        "單次調倉換手率上限 (%)",
        min_value=10.0,
        max_value=100.0,
        value=float(cfg.get("max_turnover", 0.40)) * 100.0,
        step=5.0,
    )

    with st.expander("🏛️ 港交所規費與市場衝擊滑價參數", expanded=False):
        broker_comm_bps = st.number_input(
            "經紀佣金率 (% 費率，預設 0.03%)",
            min_value=0.0,
            max_value=0.50,
            value=float(cfg.get("broker_comm_rate", DEFAULT_BROKER_COMM_RATE)) * 100.0,
            step=0.01,
            format="%.3f",
        )
        min_broker_comm = st.number_input(
            "最低經紀佣金 (HKD，預設 15 元)",
            min_value=0.0,
            max_value=200.0,
            value=float(cfg.get("min_broker_comm_hkd", DEFAULT_MIN_BROKER_COMM_HKD)),
            step=5.0,
        )
        almgren_gamma = st.slider(
            "Almgren-Chriss 衝擊係數 γ",
            min_value=0.10,
            max_value=1.00,
            value=float(cfg.get("almgren_gamma", DEFAULT_ALMGREN_GAMMA)),
            step=0.05,
        )
        min_adv_input = st.number_input(
            "最低 20 日均成交額門檻 (HKD)",
            min_value=0,
            value=int(cfg.get("min_adv_hkd", 5000000)),
            step=1000000,
        )
        num_trials = st.number_input(
            "回測多重檢定次數 (DSR K)",
            min_value=1,
            max_value=500,
            value=int(cfg.get("num_backtest_trials", 5)),
            step=1,
        )

    # Reactive update & automatic atomic persistence whenever parameters change
    cfg["total_capital_hkd"] = float(total_capital_hkd)
    cfg["risk_free_rate"] = float(rf_rate_pct) / 100.0
    cfg["max_single_position_pct"] = float(max_pos_pct) / 100.0
    cfg["max_turnover"] = float(max_turnover_pct) / 100.0
    cfg["min_adv_hkd"] = float(min_adv_input)
    cfg["alloc_method"] = alloc_method
    cfg["cov_estimator"] = cov_estimator
    cfg["broker_comm_rate"] = float(broker_comm_bps) / 100.0
    cfg["min_broker_comm_hkd"] = float(min_broker_comm)
    cfg["almgren_gamma"] = float(almgren_gamma)
    cfg["num_backtest_trials"] = int(num_trials)

    new_hash = json.dumps(cfg, sort_keys=True, ensure_ascii=False)
    if new_hash != st.session_state.last_saved_hash:
        if save_portfolio_config(cfg):
            st.session_state.last_saved_hash = new_hash
            st.caption("🟢 參數已自動即時同步至設定檔")
        else:
            st.warning("⚠️ 設定檔寫入受阻，目前維持記憶體狀態")
    else:
        st.caption("🟢 系統狀態已與設定檔完全同步")


# ==========================================
# PAGE 0: Fancy Special-Effects Portal ("香港股市預測")
# ==========================================
macro_info = cached_macro_regime()
hsi_dist = macro_info.get(
    "hsi_dist_pct",
    (macro_info["hsi_close"] - macro_info["hsi_sma200"]) / (macro_info["hsi_sma200"] + 1e-9),
)

if not st.session_state.entered_main_terminal:
    # Dynamically compute real-time prices & calibrated multi-factor probabilities for showcase cards
    showcase_defaults = get_live_portal_showcase_cards(cfg)
    cards_html = ""
    for item in showcase_defaults:
        cards_html += f"""
        <div class="portal-stock-card">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                <span style="font-weight:700; color:#F8FAFC; font-size:0.98rem;">{item['name']}</span>
                <span class="mono-num" style="color:#94A3B8; font-size:0.80rem;">{item['code']}</span>
            </div>
            <div style="display:flex; justify-content:space-between; align-items:baseline; margin-bottom:10px;">
                <span class="mono-num" style="font-size:1.45rem; color:#FFFFFF;">${item['price']:.2f}</span>
                <span style="font-size:0.78rem; padding:3px 9px; border-radius:999px; background:rgba(56,189,248,0.12); color:{item['color']}; border:1px solid {item['color']}55; font-weight:600;">
                    {item['tag']}
                </span>
            </div>
            <div style="display:flex; justify-content:space-between; font-size:0.78rem; color:#94A3B8; margin-bottom:4px;">
                <span>AI 上漲機率預測</span>
                <span class="mono-num" style="color:{item['color']};">{item['prob']:.1f}% ({item['z']:+.2f}σ)</span>
            </div>
            <div class="factor-track">
                <div style="height:100%; width:{min(100.0, max(10.0, item['prob'])):.1f}%; background:linear-gradient(90deg, #0EA5E9, {item['color']}); border-radius:999px;"></div>
            </div>
        </div>
        """
    hsi_prob_pct = macro_info.get("macro_prob", 0.5) * 100.0
    st.markdown(
        f"""
        <div class="hk-portal-stage">
            <div class="portal-eyebrow">
                <span class="live-dot"></span>
                <span>HKEX AI QUANTITATIVE PREDICTION ENGINE · 機構級多因子量化平台</span>
            </div>
            <div class="portal-main-title">香港股市預測</div>
            <div class="portal-subtitle">
                結合<strong>無未來函數統計校準多因子模型</strong>、<strong>港交所真實微觀結構與市場衝擊滑價</strong>，以及<strong>短中長線全週期歷史勝率檢驗</strong>，將複雜的機構量化分析化繁為簡，為您呈現清晰、直觀、零程式碼干擾的港股智慧決策體驗。
            </div>
            <div style="display:flex; justify-content:center; flex-wrap:wrap; gap:18px; margin-bottom:26px;">
                <div style="background:rgba(15,23,42,0.85); border:1px solid rgba(56,189,248,0.3); border-radius:12px; padding:10px 20px;">
                    <span style="color:#94A3B8; font-size:0.82rem;">恆生指數 HSI：</span>
                    <span class="mono-num" style="font-size:1.1rem; color:#F8FAFC;">{macro_info['hsi_close']:,.0f}</span>
                    <span class="mono-num" style="color:{'#34D399' if hsi_dist>=0 else '#FB7185'}; font-size:0.86rem;"> ({hsi_dist*100:+.1f}% vs 牛熊線)</span>
                </div>
                <div style="background:rgba(15,23,42,0.85); border:1px solid rgba(16,185,129,0.3); border-radius:12px; padding:10px 20px;">
                    <span style="color:#94A3B8; font-size:0.82rem;">大盤多頭情緒溫度：</span>
                    <span class="mono-num" style="font-size:1.1rem; color:#34D399;">{hsi_prob_pct:.1f}%</span>
                </div>
                <div style="background:rgba(15,23,42,0.85); border:1px solid rgba(168,85,247,0.3); border-radius:12px; padding:10px 20px;">
                    <span style="color:#94A3B8; font-size:0.82rem;">當前選中分析標的：</span>
                    <span class="mono-num" style="font-size:1.1rem; color:#38BDF8;">{active_ticker}</span>
                </div>
            </div>
            <div class="portal-orb-row">
                {cards_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    btn_c1, btn_c2, btn_c3 = st.columns([1.0, 1.4, 1.0])
    with btn_c2:
        if st.button("🚀 立即進入「香港股市預測」四大分析工作台", use_container_width=True, type="primary"):
            st.session_state.entered_main_terminal = True
            st.rerun()

    # Clean, easy-to-understand 4-pillar feature overview
    g1, g2, g3, g4 = st.columns(4)
    with g1:
        st.markdown(
            """
            <div class="bento-card">
                <div class="bento-title"><span>⚡ 01 · 日內當沖戰術</span><span style="color:#38BDF8;">即時買賣點</span></div>
                <div style="font-size:0.86rem; color:#CBD5E1; line-height:1.55;">
                    嚴格區分「上一根確認收盤價」與「次根開盤實際成交價」，每日 09:30 重置日內均價線，並自動計算港交所印花稅與最佳止損止盈位。
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with g2:
        st.markdown(
            """
            <div class="bento-card">
                <div class="bento-title"><span>🔍 02 · 趨勢機率透視</span><span style="color:#34D399;">多因子校準</span></div>
                <div style="font-size:0.86rem; color:#CBD5E1; line-height:1.55;">
                    捨棄主觀硬編碼加減分，改以統計機率模型輸出 0%–100% 上漲機率與四大核心因子（動量、價值、低波、流動性）獨立貢獻度。
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with g3:
        st.markdown(
            """
            <div class="bento-card">
                <div class="bento-title"><span>📊 03 · 穩健組合調倉</span><span style="color:#F59E0B;">去噪與風控</span></div>
                <div style="font-size:0.86rem; color:#CBD5E1; line-height:1.55;">
                    內建停牌遮罩與矩陣去噪技術，自動排除上市未滿 120 日之新股，生成符合港股每手股數與大額拆單提示的實盤調倉單。
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with g4:
        st.markdown(
            """
            <div class="bento-card">
                <div class="bento-title"><span>🎯 04 · 歷史勝率檢驗</span><span style="color:#A855F7;">全週期回測</span></div>
                <div style="font-size:0.86rem; color:#CBD5E1; line-height:1.55;">
                    自由切換未來 5 日、10 日、1 個月、3 個月、半年至 1 年預測週期，一鍵檢視歷史方向勝率、五分位數收益階梯與假陽性風險。
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.stop()


# ==========================================
# Top Macro Regime Live Status Ribbon (Main Terminal)
# ==========================================
regime_badge = (
    '<span class="badge-bull">BULL REGIME 牛市進攻</span>'
    if not macro_info["bear_market"]
    else '<span class="badge-bear">DEFENSIVE 熊市防禦</span>'
)

st.markdown(
    f"""
<div class="macro-ribbon">
    <div class="macro-brand">
        <span class="live-dot"></span>
        <span>香港股市預測 · HKEX ALPHA ENGINE</span>
        <span style="color: #38BDF8; font-weight: 500; font-size: 0.80rem;">INSTITUTIONAL TERMINAL</span>
    </div>
    <div class="macro-pills">
        <div class="macro-item">
            <span>恆生指數 HSI:</span>
            <span class="mono-num">{macro_info['hsi_close']:,.0f}</span>
            <span class="mono-num" style="color: {'#34D399' if hsi_dist >= 0 else '#FB7185'};">({hsi_dist*100:+.1f}% vs 200MA)</span>
        </div>
        <div class="macro-item">
            <span>US 10Y 殖利率:</span>
            <span class="mono-num">{macro_info['tnx_yield']:.2f}%</span>
        </div>
        <div class="macro-item">
            <span>宏觀閘門:</span>
            {regime_badge}
        </div>
        <div class="macro-item">
            <span>當前追蹤:</span>
            <span class="mono-num" style="color: #38BDF8;">{active_ticker} ({interval_choice})</span>
        </div>
        <div class="macro-item">
            <span>去噪與配置:</span>
            <span class="mono-num">{alloc_method.upper()} + {cov_estimator.upper()}</span>
        </div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)


# ==========================================
# Main Workspace Tabs (4 Clean, Visual Pages)
# ==========================================
tab1, tab2, tab3, tab4 = st.tabs([
    f"⚡ 日內當沖與 VWAP 戰術 ({active_ticker})",
    f"🔍 雙時段趨勢機率與因子透視 ({active_ticker})",
    "📊 穩健組合最佳化與實盤調倉工作台",
    f"🎯 歷史勝率與多週期預測檢驗 ({active_ticker})",
])


# ==========================================
# TAB 1: Intraday Day Trading & Microstructure
# ==========================================
with tab1:
    with st.spinner(f"正在計算 {active_ticker} ({interval_choice}) 無未來函數訊號與港交所微觀滑價..."):
        day_res, day_err = cached_day_trading_signal(
            active_ticker,
            float(min_adv_input),
            interval_choice,
            float(cfg["broker_comm_rate"]),
            float(cfg["min_broker_comm_hkd"]),
            float(cfg["almgren_gamma"]),
        )

    if day_err or not day_res:
        st.error(f"無法載入 {active_ticker}：{day_err}")
    else:
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("最新實盤現價", f"${day_res['price']:.2f}", f"決策價 ${day_res['decision_price']:.2f}")
        m2.metric("次根開盤預估撮合價", f"${day_res['execution_price']:.2f}", f"含滑價 {day_res['microstructure']['slippage_bps']:.1f} bps")
        m3.metric("AI 上漲趨勢機率", f"{day_res['up_prob']*100:.1f}%", f"合成 Z {day_res['composite_zscore']:+.2f}σ")
        m4.metric("日內 VWAP 均價 (09:30重置)", f"${day_res['vwap']:.2f}", f"{((day_res['decision_price']-day_res['vwap'])/day_res['vwap'])*100:+.2f}%")
        m5.metric("時段標準化量能分位", f"P{day_res['vol_pct_rank']:.0f}", f"{day_res['vol_ratio']:.2f}x 同刻均量")
        m6.metric(f"每手入場市值 ({day_res['lot_size']}股)", f"${day_res['lot_cost_hkd']:,.0f}", f"法定 Tick ${day_res['tick_size']}")

        b_left, b_right = st.columns([1.1, 1.2])

        with b_left:
            prob_val = day_res["up_prob"]
            sig_badge_cls = "badge-bull" if prob_val >= 0.65 else ("badge-bear" if prob_val <= 0.35 else "badge-warn")
            sfc_badge = (
                '<span class="badge-bull">SFC 可沽空名單</span>'
                if day_res["short_eligible"]
                else '<span class="badge-warn">非可沽空 / 僅限做多</span>'
            )
            mc = day_res["microstructure"]
            st.markdown(
                f"""
            <div class="bento-card">
                <div class="bento-title">
                    <span>🎯 無未來函數決策點位與港交所真實規費試算</span>
                    {sfc_badge}
                </div>
                <div style="margin: 8px 0 14px 0;">
                    <span class="{sig_badge_cls}" style="font-size: 0.90rem; padding: 5px 12px;">
                        {day_res['action']}
                    </span>
                </div>
                <div class="exec-grid">
                    <div class="exec-cell">
                        <div class="exec-label">訊號決策價 (t-1收盤) / 實盤撮合價 (次根開盤)</div>
                        <div class="exec-val">${day_res['decision_price']:.2f} <span style="font-size:0.84rem; color:#38BDF8;">➔ ${day_res['execution_price']:.2f}</span></div>
                    </div>
                    <div class="exec-cell">
                        <div class="exec-label">日內 VWAP 均價 / Wilder's RSI (14)</div>
                        <div class="exec-val" style="color:#F59E0B;">${day_res['vwap']:.2f} <span style="font-size:0.84rem; color:#F8FAFC;">| RSI {day_res['rsi']:.1f}</span></div>
                    </div>
                    <div class="exec-cell">
                        <div class="exec-label">ATR 止損價 (以實際撮合價計算並對齊 Tick)</div>
                        <div class="exec-val" style="color: #FB7185;">${day_res['stop_loss_price']:.2f}</div>
                    </div>
                    <div class="exec-cell">
                        <div class="exec-label">目標止盈價 (1:2 風險報酬比對齊 Tick)</div>
                        <div class="exec-val" style="color: #34D399;">${day_res['take_profit_price']:.2f}</div>
                    </div>
                </div>
                <div style="margin-top:12px; padding-top:10px; border-top:1px solid rgba(148,163,184,0.12); font-size:0.80rem; color:#94A3B8; display:flex; justify-content:space-between; flex-wrap:wrap; gap:8px;">
                    <span>印花稅(向上取整): <strong class="mono-num">${mc['stamp_duty_hkd']:.0f}</strong></span>
                    <span>證監+會財+交易所費: <strong class="mono-num">${mc['sfc_levy_hkd']+mc['afrc_levy_hkd']+mc['hkex_fee_hkd']:.2f}</strong></span>
                    <span>經紀佣金: <strong class="mono-num">${mc['broker_commission_hkd']:.1f}</strong></span>
                    <span>衝擊滑價: <strong class="mono-num" style="color:#FBBF24;">{mc['slippage_bps']:.1f} bps</strong></span>
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with b_right:
            st.markdown(
                render_factor_bars_html(
                    title="🔬 短線五維標準化因子暴露 (Standardized Z-Scores)",
                    subtitle=f"上漲機率: {day_res['up_prob']*100:.1f}% | 合成 Z: {day_res['composite_zscore']:+.2f}σ",
                    factor_dict=day_res["factor_zscores"],
                    max_scale=3.0,
                    unit="σ",
                ),
                unsafe_allow_html=True,
            )

        df_d = day_res["history"]
        fig_3p = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.62, 0.18, 0.20],
            subplot_titles=(
                f"{day_res['ticker']} 價格走勢、每日 09:30 重置之日內 VWAP 與 ±1.5 ATR 波動通道 ({day_res['interval']})",
                "時段成交量與 20 期均量",
                "Wilder's RSI (14) 動能強弱",
            ),
        )

        fig_3p.add_trace(
            go.Candlestick(
                x=df_d.index,
                open=df_d["Open"],
                high=df_d["High"],
                low=df_d["Low"],
                close=df_d["Close"],
                name="K線",
                increasing_line_color="#10B981",
                decreasing_line_color="#F43F5E",
            ),
            row=1,
            col=1,
        )

        if "VWAP" in df_d.columns:
            fig_3p.add_trace(
                go.Scatter(
                    x=df_d.index,
                    y=df_d["VWAP"],
                    mode="lines",
                    name="VWAP 均價線 (09:30重置)",
                    line=dict(color="#F59E0B", width=2.2),
                ),
                row=1,
                col=1,
            )

        if "ATR_Upper" in df_d.columns and "ATR_Lower" in df_d.columns:
            fig_3p.add_trace(
                go.Scatter(
                    x=df_d.index,
                    y=df_d["ATR_Upper"],
                    mode="lines",
                    name="+1.5 ATR 通道",
                    line=dict(color="rgba(56, 189, 248, 0.35)", width=1, dash="dot"),
                ),
                row=1,
                col=1,
            )
            fig_3p.add_trace(
                go.Scatter(
                    x=df_d.index,
                    y=df_d["ATR_Lower"],
                    mode="lines",
                    name="-1.5 ATR 通道",
                    line=dict(color="rgba(56, 189, 248, 0.35)", width=1, dash="dot"),
                    fill="tonexty",
                    fillcolor="rgba(56, 189, 248, 0.05)",
                ),
                row=1,
                col=1,
            )

        vol_colors = np.where(df_d["Close"] >= df_d["Open"], "#10B981", "#F43F5E")
        fig_3p.add_trace(
            go.Bar(x=df_d.index, y=df_d["Volume"], name="成交量", marker_color=vol_colors, opacity=0.85),
            row=2,
            col=1,
        )
        if "Vol_MA20" in df_d.columns:
            fig_3p.add_trace(
                go.Scatter(
                    x=df_d.index,
                    y=df_d["Vol_MA20"],
                    mode="lines",
                    name="20期均量",
                    line=dict(color="#38BDF8", width=1.4),
                ),
                row=2,
                col=1,
            )

        if "RSI" in df_d.columns:
            fig_3p.add_trace(
                go.Scatter(
                    x=df_d.index,
                    y=df_d["RSI"],
                    mode="lines",
                    name="RSI (14)",
                    line=dict(color="#A855F7", width=2),
                ),
                row=3,
                col=1,
            )
            fig_3p.add_hline(y=70, line_dash="dash", line_color="rgba(244,63,94,0.5)", row=3, col=1)
            fig_3p.add_hline(y=30, line_dash="dash", line_color="rgba(16,185,129,0.5)", row=3, col=1)

        fig_3p.update_layout(
            height=620,
            xaxis_rangeslider_visible=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(11,15,25,0.72)",
            font=dict(color="#CBD5E1", size=11),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=15, r=15, t=35, b=15),
        )
        fig_3p.update_xaxes(showgrid=True, gridcolor="rgba(148,163,184,0.07)")
        if day_res["interval"] in ("5m", "15m"):
            fig_3p.update_xaxes(
                rangebreaks=[
                    dict(bounds=["sat", "mon"]),
                    dict(bounds=[16, 9.5], pattern="hour"),
                    dict(bounds=[12, 13], pattern="hour"),
                ]
            )
        fig_3p.update_yaxes(showgrid=True, gridcolor="rgba(148,163,184,0.07)")
        st.plotly_chart(fig_3p, use_container_width=True)

        # Level 2 Order Book & Market Depth Component (Bid 1~10 & Ask 1~10)
        ob_res = cached_order_book(
            ticker=active_ticker,
            host=cfg.get("futu_host", "127.0.0.1"),
            port=int(cfg.get("futu_port", 11111)),
            spot_price=float(day_res["price"]),
            lot_size=int(day_res["lot_size"]),
        )

        ob_badge_color = "#34D399" if ob_res["is_live"] else "#94A3B8"
        obi_color = "#34D399" if ob_res["order_book_imbalance"] >= 0 else "#FB7185"
        obi_text = "買盤佔優" if ob_res["order_book_imbalance"] >= 0 else "賣盤佔優"

        st.markdown(f"""
        <div class="bento-card" style="margin-top: 14px;">
            <div class="bento-title">
                <span>🏛️ 港股 Level 2 買賣十檔擺盤深度與微觀價差 (Market Depth 10 Levels)</span>
                <span class="mono-num" style="color: {ob_badge_color}; font-size: 0.78rem;">
                    {ob_res["source"]}
                </span>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px; margin-bottom: 12px; font-size: 0.82rem; color: #94A3B8;">
                <div>最優買賣價差: <strong class="mono-num" style="color: #F8FAFC;">${ob_res["spread"]:.3f}</strong> ({ob_res["spread_ticks"]} Tick / {ob_res["spread_bps"]:.1f} bps)</div>
                <div>微觀中間價 (Micro-Price): <strong class="mono-num" style="color: #38BDF8;">${ob_res["micro_price"]:.3f}</strong></div>
                <div>盤口力量不平衡 (OBI): <strong class="mono-num" style="color: {obi_color};">{ob_res["order_book_imbalance"]*100:+.1f}% ({obi_text})</strong></div>
                <div>行情接收時間: <strong class="mono-num">{ob_res["update_time"]}</strong></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        ob_col1, ob_col2 = st.columns(2)
        max_vol_scale = max(1, max(b["volume"] for b in ob_res["bids"] + ob_res["asks"]))
        with ob_col1:
            st.markdown("##### 🟢 委買十檔 (Bids 1~10)")
            df_bids = pd.DataFrame(ob_res["bids"])
            df_bids["檔位"] = [f"買 {i}" for i in df_bids["level"]]
            st.dataframe(
                df_bids[["檔位", "price", "volume", "orders"]],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "檔位": st.column_config.TextColumn("檔位"),
                    "price": st.column_config.NumberColumn("委買價", format="$%.3f"),
                    "volume": st.column_config.ProgressColumn("委買量 (股)", format="%d", min_value=0, max_value=max_vol_scale),
                    "orders": st.column_config.NumberColumn("筆數", format="%d 筆"),
                },
            )
        with ob_col2:
            st.markdown("##### 🔴 委賣十檔 (Asks 1~10)")
            df_asks = pd.DataFrame(ob_res["asks"])
            df_asks["檔位"] = [f"賣 {i}" for i in df_asks["level"]]
            st.dataframe(
                df_asks[["檔位", "price", "volume", "orders"]],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "檔位": st.column_config.TextColumn("檔位"),
                    "price": st.column_config.NumberColumn("委賣價", format="$%.3f"),
                    "volume": st.column_config.ProgressColumn("委賣量 (股)", format="%d", min_value=0, max_value=max_vol_scale),
                    "orders": st.column_config.NumberColumn("筆數", format="%d 筆"),
                },
            )

        with st.expander("💡 如何啟用富途牛牛 FutuOpenD 毫秒級實盤串流？", expanded=False):
            st.markdown("""
            1. **下載安裝 FutuOpenD**：前往[富途官方開發者中心](https://openapi.futunn.com/)下載安裝 FutuOpenD 閘道程式並登入富途帳號。
            2. **確認監聽位址**：預設監聽位址為 `127.0.0.1:11111`（可於左側側邊欄自由修改）。
            3. **安裝 Python 套件**：在終端機執行 `pip install futu-api`。
            4. **一鍵切換**：在側邊欄開啟「啟用富途 Level 2 串流報價」並測試連線，系統將自動切換為港股 Level 2 毫秒級實盤串流！
            """)


# ==========================================
# TAB 2: Calibrated Multi-Factor Probability & VIF Matrix
# ==========================================
with tab2:
    with st.spinner(f"正在計算 {active_ticker} 的 IC 加權多因子機率模型與共線性診斷..."):
        trend_res, trend_err = cached_stock_trend(active_ticker)

    if trend_err or not trend_res:
        st.error(f"無法分析 {active_ticker}：{trend_err}")
    else:
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("最新現價 / 次日撮合價", f"${trend_res['price']:.2f}", f"撮合 ${trend_res['execution_price']:.2f}")
        c2.metric("趨勢上漲機率 P(Up)", f"{trend_res['up_prob']*100:.1f}%", f"合成 Z {trend_res['composite_zscore']:+.2f}σ")
        c3.metric("短期戰術上漲機率", f"{trend_res['short_term_prob']*100:.1f}%", f"20D Z {trend_res['z_20d']:+.2f}σ")
        c4.metric("長期戰略上漲機率", f"{trend_res['long_term_prob']*100:.1f}%", f"200MA {trend_res['dist_200ma']*100:+.1f}%")
        c5.metric("滾動股息殖利率", f"{trend_res['dividend_yield']*100:.2f}%", "前復權已計入")
        c6.metric("股東權益報酬 (ROE)", f"{trend_res['roe']*100:.1f}%", "財報質量因子")

        t2_left, t2_right = st.columns([1.15, 1.15])
        with t2_left:
            z_dict_display = {
                row["factor_name"]: float(row["z_score"])
                for row in trend_res["factor_exposure_matrix"]
            }
            st.markdown(
                render_factor_bars_html(
                    title="🏛️ 資訊係數 (IC) 加權四大核心因子標準化暴露",
                    subtitle=f"{trend_res['trend']}",
                    factor_dict=z_dict_display,
                    max_scale=3.0,
                    unit="σ",
                ),
                unsafe_allow_html=True,
            )
        with t2_right:
            bullets_html = "".join(
                f'<div style="padding: 6px 0; border-bottom: 1px solid rgba(148,163,184,0.08); font-size: 0.84rem; color: #CBD5E1;">{r}</div>'
                for r in trend_res["reasons"]
            )
            st.markdown(
                f"""
            <div class="bento-card">
                <div class="bento-title">
                    <span>📋 邏輯回歸趨勢機率與無未來函數診斷明細</span>
                    <span class="mono-num" style="color:#34D399;">P(Up): {trend_res['up_prob']*100:.1f}%</span>
                </div>
                {bullets_html}
            </div>
            """,
                unsafe_allow_html=True,
            )

        # Factor Exposure & Collinearity Diagnostics Matrix
        exp_col, corr_col = st.columns([1.35, 0.95])
        with exp_col:
            st.markdown("#### 📐 標準化因子暴露矩陣與共線性診斷 (VIF 變異數膨脹因子)")
            df_exp = pd.DataFrame(trend_res["factor_exposure_matrix"])
            st.dataframe(
                df_exp[[
                    "factor_name",
                    "z_score",
                    "rank_ic",
                    "ic_weight_pct",
                    "marginal_contribution_z",
                    "vif",
                    "collinearity_status",
                ]],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "factor_name": st.column_config.TextColumn("核心因子名稱"),
                    "z_score": st.column_config.NumberColumn("標準化 Z-Score", format="%+.2fσ"),
                    "rank_ic": st.column_config.NumberColumn("滾動 Rank IC", format="%+.3f"),
                    "ic_weight_pct": st.column_config.ProgressColumn("IC 校準權重", format="%.1f%%", min_value=0, max_value=100),
                    "marginal_contribution_z": st.column_config.NumberColumn("邊際貢獻度", format="%+.3fσ"),
                    "vif": st.column_config.NumberColumn("共線性 VIF", format="%.2f"),
                    "collinearity_status": st.column_config.TextColumn("正交診斷結果"),
                },
            )
        with corr_col:
            st.markdown("#### 🔗 因子間相關性診斷矩陣")
            f_corr = trend_res["factor_corr_matrix"].copy()
            short_names = ["動量", "價值質量", "低波品質", "流動資金"]
            f_corr.index = short_names
            f_corr.columns = short_names
            fig_fcorr = px.imshow(
                f_corr, text_auto=".2f", color_continuous_scale="Tealrose", zmin=-1, zmax=1
            )
            fig_fcorr.update_layout(
                height=245,
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#CBD5E1", size=11),
                margin=dict(l=10, r=10, t=10, b=10),
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig_fcorr, use_container_width=True)

        df_h = trend_res["history"]
        fig_t2 = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.04,
            row_heights=[0.72, 0.28],
            subplot_titles=(
                f"{trend_res['ticker']} 前復權股價走勢 + 50MA / 200MA 牛熊分界線 (消除除息跳空失真)",
                "滾動 20 日波動率標準化動量 Z-Score (±3σ 統計校準區間)",
            ),
        )
        fig_t2.add_trace(
            go.Candlestick(
                x=df_h.index,
                open=df_h["Open"],
                high=df_h["High"],
                low=df_h["Low"],
                close=df_h["Close"],
                name="前復權 K線",
                increasing_line_color="#10B981",
                decreasing_line_color="#F43F5E",
            ),
            row=1,
            col=1,
        )

        if "SMA50" in df_h.columns:
            fig_t2.add_trace(
                go.Scatter(
                    x=df_h.index,
                    y=df_h["SMA50"],
                    mode="lines",
                    name="SMA 50 (中線)",
                    line=dict(color="#38BDF8", width=1.8),
                ),
                row=1,
                col=1,
            )
        if "SMA200" in df_h.columns:
            fig_t2.add_trace(
                go.Scatter(
                    x=df_h.index,
                    y=df_h["SMA200"],
                    mode="lines",
                    name="SMA 200 (牛熊線)",
                    line=dict(color="#F59E0B", width=2.2),
                ),
                row=1,
                col=1,
            )

        if "Z_20D" in df_h.columns:
            z_colors = np.where(df_h["Z_20D"] >= 0, "#10B981", "#F43F5E")
            fig_t2.add_trace(
                go.Bar(
                    x=df_h.index,
                    y=df_h["Z_20D"],
                    name="20日動量 Z-Score",
                    marker_color=z_colors,
                    opacity=0.85,
                ),
                row=2,
                col=1,
            )
            fig_t2.add_hline(y=1.5, line_dash="dot", line_color="rgba(16,185,129,0.5)", row=2, col=1)
            fig_t2.add_hline(y=-1.5, line_dash="dot", line_color="rgba(244,63,94,0.5)", row=2, col=1)

        fig_t2.update_layout(
            height=540,
            xaxis_rangeslider_visible=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(11,15,25,0.72)",
            font=dict(color="#CBD5E1", size=11),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=15, r=15, t=35, b=15),
        )
        fig_t2.update_xaxes(showgrid=True, gridcolor="rgba(148,163,184,0.07)")
        fig_t2.update_yaxes(showgrid=True, gridcolor="rgba(148,163,184,0.07)")
        st.plotly_chart(fig_t2, use_container_width=True)


# ==========================================
# TAB 3: Portfolio Workbench, Denoised Cov & Blotter
# ==========================================
with tab3:
    st.markdown("""
    <div class="bento-card" style="padding: 14px 20px; margin-bottom: 14px;">
        <div class="bento-title" style="margin-bottom: 4px;">
            <span>💼 投資組合持倉管理與多標的配置工作台</span>
            <span style="color: #38BDF8;">支援點選預設範本、一鍵加入任意港股，或雙擊下方表格直接修改</span>
        </div>
        <div style="font-size:0.84rem; color:#94A3B8; margin-top:4px;">
            💡 <strong>如何更換持倉組合？</strong> 您可以：① 點選下方 4 組<strong>精選範本</strong>一鍵切換；② 在輸入框輸入代號（如 1810.HK）並點擊<strong>加入持倉</strong>；③ 展開下方表格<strong>直接雙擊代號或股數修改</strong>，或勾選刪除。修改將自動即時同步。
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("##### 🚀 快速套用精選投資組合範本 (一鍵切換)")
    p_col1, p_col2, p_col3, p_col4 = st.columns(4)
    with p_col1:
        if st.button("🌐 港股科技巨頭\n(0700, 9988, 3690, 1810)", use_container_width=True):
            cfg["holdings"] = [
                {"ticker": "0700.HK", "name": "騰訊控股", "lot_size": 100, "shares": 300, "entry_price": 380.0, "short_eligible": True},
                {"ticker": "9988.HK", "name": "阿里巴巴-W", "lot_size": 100, "shares": 500, "entry_price": 85.0, "short_eligible": True},
                {"ticker": "3690.HK", "name": "美團-W", "lot_size": 100, "shares": 400, "entry_price": 130.0, "short_eligible": True},
                {"ticker": "1810.HK", "name": "小米集團-W", "lot_size": 200, "shares": 2000, "entry_price": 25.0, "short_eligible": True},
            ]
            save_portfolio_config(cfg)
            st.session_state.last_saved_hash = json.dumps(cfg, sort_keys=True, ensure_ascii=False)
            st.rerun()
    with p_col2:
        if st.button("🏛️ 藍籌金融核心\n(0005, 0388, 1299, 0939)", use_container_width=True):
            cfg["holdings"] = [
                {"ticker": "0005.HK", "name": "匯豐控股", "lot_size": 400, "shares": 1200, "entry_price": 68.0, "short_eligible": True},
                {"ticker": "0388.HK", "name": "香港交易所", "lot_size": 100, "shares": 300, "entry_price": 280.0, "short_eligible": True},
                {"ticker": "1299.HK", "name": "友邦保險", "lot_size": 200, "shares": 1000, "entry_price": 62.0, "short_eligible": True},
                {"ticker": "0939.HK", "name": "建設銀行", "lot_size": 1000, "shares": 10000, "entry_price": 5.8, "short_eligible": True},
            ]
            save_portfolio_config(cfg)
            st.session_state.last_saved_hash = json.dumps(cfg, sort_keys=True, ensure_ascii=False)
            st.rerun()
    with p_col3:
        if st.button("🚗 新能源與製造\n(1211, 2015, 0175, 2382)", use_container_width=True):
            cfg["holdings"] = [
                {"ticker": "1211.HK", "name": "比亞迪股份", "lot_size": 500, "shares": 500, "entry_price": 260.0, "short_eligible": True},
                {"ticker": "2015.HK", "name": "理想汽車-W", "lot_size": 100, "shares": 600, "entry_price": 100.0, "short_eligible": True},
                {"ticker": "0175.HK", "name": "吉利汽車", "lot_size": 1000, "shares": 4000, "entry_price": 12.0, "short_eligible": True},
                {"ticker": "2382.HK", "name": "舜宇光學", "lot_size": 100, "shares": 500, "entry_price": 55.0, "short_eligible": True},
            ]
            save_portfolio_config(cfg)
            st.session_state.last_saved_hash = json.dumps(cfg, sort_keys=True, ensure_ascii=False)
            st.rerun()
    with p_col4:
        if st.button("🛡️ 高息防禦公用\n(0941, 0883, 0002, 0003)", use_container_width=True):
            cfg["holdings"] = [
                {"ticker": "0941.HK", "name": "中國移動", "lot_size": 500, "shares": 1500, "entry_price": 72.0, "short_eligible": True},
                {"ticker": "0883.HK", "name": "中國海洋石油", "lot_size": 1000, "shares": 3000, "entry_price": 18.0, "short_eligible": True},
                {"ticker": "0002.HK", "name": "中電控股", "lot_size": 500, "shares": 1500, "entry_price": 65.0, "short_eligible": True},
                {"ticker": "0003.HK", "name": "中華煤氣", "lot_size": 1000, "shares": 5000, "entry_price": 6.2, "short_eligible": True},
            ]
            save_portfolio_config(cfg)
            st.session_state.last_saved_hash = json.dumps(cfg, sort_keys=True, ensure_ascii=False)
            st.rerun()

    qa_c1, qa_c2, qa_c3 = st.columns([1.5, 1.0, 1.0])
    with qa_c1:
        add_ticker_input = st.text_input("➕ 快速新增標的代號 (例如 1810.HK 或 0941)", placeholder="輸入港股代號如 1810.HK", key="quick_add_input")
    with qa_c2:
        add_shares_input = st.number_input("持有股數", min_value=0, value=500, step=100, key="quick_add_shares")
    with qa_c3:
        st.write("")
        st.write("")
        if st.button("加入持倉組合", type="secondary", use_container_width=True):
            if add_ticker_input.strip():
                norm_add = normalize_hk_ticker(add_ticker_input.strip())
                curr_tickers = [h.get("ticker", "") for h in cfg.get("holdings", [])]
                if norm_add not in curr_tickers:
                    cfg["holdings"].append({
                        "ticker": norm_add,
                        "name": norm_add,
                        "lot_size": HK_DEFAULT_LOT_SIZES.get(norm_add, 100),
                        "shares": int(add_shares_input),
                        "entry_price": 0.0,
                        "short_eligible": is_hk_short_eligible(norm_add),
                    })
                    save_portfolio_config(cfg)
                    st.session_state.last_saved_hash = json.dumps(cfg, sort_keys=True, ensure_ascii=False)
                    st.rerun()
                else:
                    st.toast(f"{norm_add} 已存在於持倉中！")

    with st.expander("✏️ 詳細持倉數據表 (支援雙擊儲存格修改代號/股數，或勾選刪除)", expanded=True):
        df_edit_in = pd.DataFrame(cfg.get("holdings", []))
        if df_edit_in.empty:
            df_edit_in = pd.DataFrame(columns=["ticker", "name", "lot_size", "shares"])
        else:
            for col in ["ticker", "name", "lot_size", "shares"]:
                if col not in df_edit_in.columns:
                    df_edit_in[col] = ""

        edited_df = st.data_editor(
            df_edit_in[["ticker", "name", "lot_size", "shares"]],
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "ticker": st.column_config.TextColumn("港股代號 (如 0700.HK)", required=True),
                "name": st.column_config.TextColumn("股票名稱", required=True),
                "lot_size": st.column_config.NumberColumn("每手股數 (Lot Size)", min_value=1, step=50, required=True),
                "shares": st.column_config.NumberColumn("目前持有股數", min_value=0, step=100, required=True),
            },
            key="holdings_editor",
        )

        existing_map = {normalize_hk_ticker(h.get("ticker", "")): h for h in cfg.get("holdings", [])}
        new_holdings = []
        for _, row in edited_df.iterrows():
            t_str = str(row.get("ticker", "")).strip()
            if not t_str:
                continue
            norm_tk = normalize_hk_ticker(t_str)
            base_item = dict(existing_map.get(norm_tk, {}))
            base_item.update({
                "ticker": norm_tk,
                "name": str(row.get("name", norm_tk)).strip(),
                "lot_size": int(row.get("lot_size") or HK_DEFAULT_LOT_SIZES.get(norm_tk, 100)),
                "shares": int(row.get("shares") or 0),
            })
            new_holdings.append(base_item)

        if new_holdings != cfg.get("holdings", []):
            cfg["holdings"] = new_holdings
            if save_portfolio_config(cfg):
                st.session_state.last_saved_hash = json.dumps(cfg, sort_keys=True, ensure_ascii=False)

    holdings_serialized = json.dumps(cfg["holdings"], ensure_ascii=False, sort_keys=True)

    with st.spinner("正在執行停牌遮罩過濾、Ledoit-Wolf / RMT 矩陣去噪與無未來函數滾動回測..."):
        health_res = cached_portfolio_health(
            holdings_serialized,
            float(total_capital_hkd),
            float(rf_rate_pct) / 100.0,
            float(max_pos_pct) / 100.0,
            float(min_adv_input),
            alloc_method,
            int(num_trials),
            cov_estimator,
            float(max_turnover_pct) / 100.0,
            float(cfg["broker_comm_rate"]),
            float(cfg["min_broker_comm_hkd"]),
            float(cfg["almgren_gamma"]),
        )

    if not health_res:
        st.error("無法計算組合風險，請確認持倉表格中的港股代號是否有效且具備至少 120 個有效交易日。")
    else:
        bt = health_res["backtest"]

        k1, k2, k3, k4, k5, k6 = st.columns(6)
        k1.metric("現有持倉總市值", f"${health_res['total_value']:,.0f}", f"目標池 ${total_capital_hkd:,.0f}")
        k2.metric("組合風控健康度", f"{health_res['health_score']}/100", health_res["health_status"].split(" ")[1])
        if bt:
            k3.metric("滾動回測幾何年化", f"{bt['cagr']*100:+.2f}%", f"總回報 {bt['total_return']*100:+.1f}%")
            k4.metric(f"夏普比率 (Rf={rf_rate_pct:.1f}%)", f"{bt['sharpe']:.2f}", f"波動率 {bt['ann_volatility']*100:.1f}%")
            k5.metric("DSR 抗過擬合信心度", f"{bt['dsr_prob']*100:.1f}%", f"p={bt['dsr_pvalue']:.3f}")
            k6.metric("最大回撤 / 1D CVaR", f"{bt['max_drawdown']*100:.1f}%", f"CVaR {health_res['cvar_95']*100:.2f}%")
        else:
            k3.metric("組合年化波動率", f"{health_res['ann_volatility']*100:.2f}%")
            k4.metric("現有夏普比率", f"{health_res['sharpe_ratio']:.2f}")
            k5.metric("單日尾部 CVaR (95%)", f"{health_res['cvar_95']*100:.2f}%")
            k6.metric("去噪後平均相關性", f"{health_res['avg_correlation']:.2f}")

        chart_left, chart_right = st.columns([1.35, 1.0])

        with chart_left:
            if bt:
                fig_bt = make_subplots(
                    rows=2,
                    cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.05,
                    row_heights=[0.68, 0.32],
                    subplot_titles=(
                        f"樣本外滾動回測淨值 ({method_options[alloc_method]} | 嚴格 t-1 窗口 + 港交所規費與衝擊滑價)",
                        "水下回撤曲線 (Underwater Drawdown %)",
                    ),
                )
                fig_bt.add_trace(
                    go.Scatter(
                        x=bt["dates"],
                        y=bt["strategy_nav"],
                        mode="lines",
                        name=f"{alloc_method.upper()} 策略淨值",
                        line=dict(color="#10B981", width=2.4),
                    ),
                    row=1,
                    col=1,
                )
                fig_bt.add_trace(
                    go.Scatter(
                        x=bt["dates"],
                        y=bt["benchmark_nav"],
                        mode="lines",
                        name="等權重基準 (Benchmark)",
                        line=dict(color="#94A3B8", width=1.5, dash="dash"),
                    ),
                    row=1,
                    col=1,
                )

                fig_bt.add_trace(
                    go.Scatter(
                        x=bt["dates"],
                        y=bt["drawdown_series"] * 100.0,
                        mode="lines",
                        name="策略水下回撤 (%)",
                        line=dict(color="#F43F5E", width=1.6),
                        fill="tozeroy",
                        fillcolor="rgba(244, 63, 94, 0.18)",
                    ),
                    row=2,
                    col=1,
                )

                fig_bt.update_layout(
                    height=460,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(11,15,25,0.72)",
                    font=dict(color="#CBD5E1", size=11),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    margin=dict(l=15, r=15, t=35, b=15),
                )
                fig_bt.update_xaxes(showgrid=True, gridcolor="rgba(148,163,184,0.07)")
                fig_bt.update_yaxes(showgrid=True, gridcolor="rgba(148,163,184,0.07)")
                st.plotly_chart(fig_bt, use_container_width=True)

        with chart_right:
            sub_t1, sub_t2 = st.tabs(["⚖️ 當前 vs 目標權重 (含換手約束)", "🔥 去噪後資產相關性矩陣"])
            with sub_t1:
                curr_w = health_res["weights"]
                opt_w = health_res["optimal_weights"]
                tickers_all = list(opt_w.keys())
                df_w = pd.DataFrame({
                    "Ticker": tickers_all,
                    "當前權重 (%)": [curr_w.get(t, 0.0) * 100.0 for t in tickers_all],
                    "目標權重 (%)": [opt_w.get(t, 0.0) * 100.0 for t in tickers_all],
                })
                fig_w = go.Figure()
                fig_w.add_trace(
                    go.Bar(
                        y=df_w["Ticker"],
                        x=df_w["當前權重 (%)"],
                        orientation="h",
                        name="當前持倉權重",
                        marker_color="#64748B",
                    )
                )
                fig_w.add_trace(
                    go.Bar(
                        y=df_w["Ticker"],
                        x=df_w["目標權重 (%)"],
                        orientation="h",
                        name=f"{alloc_method.upper()} 目標權重",
                        marker_color="#38BDF8",
                    )
                )
                fig_w.update_layout(
                    barmode="group",
                    height=395,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(11,15,25,0.72)",
                    font=dict(color="#CBD5E1", size=11),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    margin=dict(l=10, r=10, t=25, b=10),
                )
                st.plotly_chart(fig_w, use_container_width=True)

            with sub_t2:
                corr_df = health_res["corr_matrix"]
                fig_heat = px.imshow(
                    corr_df, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1
                )
                fig_heat.update_layout(
                    height=395,
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#CBD5E1", size=11),
                    margin=dict(l=10, r=10, t=20, b=10),
                    coloraxis_showscale=False,
                )
                st.plotly_chart(fig_heat, use_container_width=True)

        blotter_col, diag_col = st.columns([1.55, 0.85])

        with blotter_col:
            st.markdown("#### 🛒 港交所實盤調倉執行單 (含法定規費、Almgren-Chriss 滑價與 >5% ADV 拆單提示)")
            exec_df = pd.DataFrame(health_res["execution_orders"])
            exec_df["target_weight_pct"] = exec_df["target_weight"] * 100.0
            st.dataframe(
                exec_df[[
                    "ticker",
                    "name",
                    "price",
                    "lot_size",
                    "target_weight_pct",
                    "current_shares",
                    "target_shares",
                    "action",
                    "estimated_turnover",
                    "statutory_fees_hkd",
                    "slippage_bps",
                    "total_friction_hkd",
                ]],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "ticker": st.column_config.TextColumn("代號"),
                    "name": st.column_config.TextColumn("名稱"),
                    "price": st.column_config.NumberColumn("Tick對齊現價", format="$%.2f"),
                    "lot_size": st.column_config.NumberColumn("每手股數", format="%d"),
                    "target_weight_pct": st.column_config.ProgressColumn(
                        "目標權重", format="%.1f%%", min_value=0, max_value=100
                    ),
                    "current_shares": st.column_config.NumberColumn("當前股數", format="%d"),
                    "target_shares": st.column_config.NumberColumn("目標股數", format="%d"),
                    "action": st.column_config.TextColumn("實盤調倉指令與拆單建議"),
                    "estimated_turnover": st.column_config.NumberColumn("預估交易額", format="$%,.0f"),
                    "statutory_fees_hkd": st.column_config.NumberColumn("法定規費+佣金", format="$%,.1f"),
                    "slippage_bps": st.column_config.NumberColumn("預估滑價 (bps)", format="%.1f"),
                    "total_friction_hkd": st.column_config.NumberColumn("總摩擦成本", format="$%,.1f"),
                },
            )

        with diag_col:
            warn_items_html = "".join(
                f'<div style="padding: 8px 0; border-bottom: 1px solid rgba(148,163,184,0.10); font-size: 0.84rem; color: #E2E8F0;">{w}</div>'
                for w in health_res["warnings"]
            )
            st.markdown(
                f"""
            <div class="bento-card" style="margin-top: 28px;">
                <div class="bento-title">
                    <span>🛡️ 機構級風控、停牌遮罩與 DSR 審計</span>
                    <span class="mono-num" style="color: #38BDF8;">{health_res['health_score']}/100</span>
                </div>
                {warn_items_html}
            </div>
            """,
                unsafe_allow_html=True,
            )


# ==========================================
# TAB 4: Historical Prediction Accuracy & Multi-Horizon Validation (Task 3)
# ==========================================
with tab4:
    st.markdown(
        """
        <div class="bento-card" style="padding: 14px 20px; margin-bottom: 16px;">
            <div class="bento-title" style="margin-bottom:4px;">
                <span>🎛️ 歷史預測準確度與多週期回測配置面板 (Short-Term to Long-Term Horizons)</span>
                <span style="color:#34D399;">嚴格 t-1 訊號決策 ➔ 次日開盤撮合 (零未來函數)</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cfg_c1, cfg_c2, cfg_c3, cfg_c4 = st.columns([1.3, 1.0, 1.0, 1.0])
    horizon_presets = {
        5: "短線戰術：未來 5 個交易日 (5D)",
        10: "短線戰術：未來 10 個交易日 (10D)",
        21: "中線波段：未來 1 個月 (21D)",
        63: "中線波段：未來 3 個月 (63D)",
        126: "長線戰略：未來 6 個月 (126D)",
        252: "長線戰略：未來 1 年 (252D)",
    }
    with cfg_c1:
        preset_k = st.selectbox(
            "選擇標準前瞻評估週期",
            options=list(horizon_presets.keys()),
            format_func=lambda k: horizon_presets[k],
            index=2,
        )
        custom_k = st.slider(
            "或自由滑動微調前瞻天數 k (交易日)",
            min_value=3,
            max_value=252,
            value=int(preset_k),
            step=1,
        )
    with cfg_c2:
        hist_len_map = {
            "6mo": "近 6 個月",
            "1y": "近 1 年",
            "3y": "近 3 年",
            "5y": "近 5 年",
        }
        hist_period_sel = st.selectbox(
            "歷史回測數據長度",
            options=list(hist_len_map.keys()),
            format_func=lambda k: hist_len_map[k],
            index=2,
        )
    with cfg_c3:
        atr_stop_mult_sel = st.slider(
            "ATR 動態停損乘數 (倍)",
            min_value=1.0,
            max_value=4.0,
            value=2.0,
            step=0.25,
        )
    with cfg_c4:
        sig_thresh_sel = st.slider(
            "訊號觸發門檻閥值 (|Z| ≥ σ)",
            min_value=0.0,
            max_value=1.5,
            value=0.25,
            step=0.05,
        )

    with st.spinner(f"正在對 {active_ticker} 進行 {hist_len_map[hist_period_sel]}、前瞻 {custom_k} 個交易日之向量化歷史準確度檢驗..."):
        acc_res, acc_err = cached_historical_accuracy(
            ticker=active_ticker,
            horizon=int(custom_k),
            history_period=hist_period_sel,
            atr_stop_mult=float(atr_stop_mult_sel),
            signal_z_threshold=float(sig_thresh_sel),
            comm_rate=float(cfg["broker_comm_rate"]),
            min_comm=float(cfg["min_broker_comm_hkd"]),
            gamma=float(cfg["almgren_gamma"]),
        )

    if acc_err or not acc_res:
        st.error(f"無法完成歷史準確度驗證：{acc_err}")
    else:
        # Row 1: 6 Core Institutional Accuracy KPI Cards
        a1, a2, a3, a4, a5, a6 = st.columns(6)
        a1.metric(
            "歷史方向勝率 (Hit Rate)",
            f"{acc_res['hit_rate']*100:.1f}%",
            f"有效訊號 {acc_res['sample_size']} 次",
        )
        a2.metric(
            "平均 Rank IC (秩相關)",
            f"{acc_res['mean_ic']:+.3f}",
            f"全樣本 IC {acc_res['overall_rank_ic']:+.3f}",
        )
        a3.metric(
            "IC 資訊比率 (IC-IR)",
            f"{acc_res['ic_ir']:+.2f}",
            "顯著預測力" if abs(acc_res["ic_ir"]) >= 0.5 else "溫和預測力",
        )
        a4.metric(
            "年化超額回報 (vs 恆指)",
            f"{acc_res['annualized_alpha']*100:+.2f}%",
            f"策略年化 {acc_res['strategy_ann_return']*100:+.1f}%",
        )
        a5.metric(
            "獲利因子 (Profit Factor)",
            f"{acc_res['profit_factor']:.2f}",
            f"盈虧比 {acc_res['win_loss_ratio']:.2f}x",
        )
        a6.metric(
            "平均實際持倉週期",
            f"{acc_res['avg_holding_days']:.1f} 日",
            f"單筆期望 {acc_res['expected_return_pct']:+.2f}%",
        )

        with st.expander("📖 機構級量化指標評級標準與判讀對照表 (點擊展開解讀各指標高低意涵)", expanded=False):
            st.markdown("""
| 核心指標 | 弱 / 警示 (Low / Poor) | 正常基準 (Normal) | 優秀 / 強健 (Good / Solid) | 極高 / 頂尖機構 (Elite) | 實戰解讀與注意事項 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **歷史方向勝率 (Hit Rate)** | `< 48.0%` (反向虧損) | `48.0% ~ 53.0%` (隨機游走) | `53.0% ~ 60.0%` (顯著優勢) | `> 62.0%` (過高需防過擬合) | 扣除印花稅與滑價後的真實勝率。盈虧比 > 1.5x 時，53%~56% 即可穩定獲利。 |
| **平均 Rank IC (秩相關)** | `< 0.015` (幾無預測力) | `0.020 ~ 0.040` (輕微預測力) | `0.040 ~ 0.080` (顯著 Alpha) | `> 0.080` (極強 Alpha，個股罕見) | 衡量評分與未來收益的秩相關。穩定維持在 0.03~0.05 即屬優質機構因子。 |
| **IC 資訊比率 (IC-IR)** | `< 0.30` (波動過大) | `0.30 ~ 0.50` (溫和可用) | `0.50 ~ 1.00` (**顯著且穩定**) | `> 1.00` (頂級穩定因子) | IC 均值 / IC 標準差。系統以 0.50 為分水嶺，≥ 0.5 標記為顯著預測力。 |
| **年化超額 (vs 恆指)** | `< 0.0%` (跑輸大盤) | `0.0% ~ 5.0%` (微幅戰勝) | `5.0% ~ 15.0%` (優秀 Alpha) | `> 20.0%` (卓越超額) | 扣除所有摩擦成本後，相對於同期恆指買入持有的淨超額年化報酬率。 |
| **獲利因子 (Profit Factor)** | `< 1.00` (總虧損 > 總盈利) | `1.00 ~ 1.35` (利潤微薄) | `1.35 ~ 2.00` (健康盈利) | `> 2.20` (極具進攻與防禦力) | 總毛利 ÷ 總毛損。大於 1.5 代表每承受 1 元虧損風險可獲取 1.5 元毛利。 |
| **盈虧比 (Win/Loss Ratio)** | `< 0.90x` (賺少賠多) | `1.00x ~ 1.40x` (均等風報) | `1.50x ~ 2.50x` (截斷虧損) | `> 2.80x` (長尾大賺小賠) | 平均每筆盈利 ÷ 平均每筆虧損。配合 ATR 止損壓低單筆虧損，保持 > 1.5x。 |
| **五分位多空差 (Q5-Q1)** | `< +1.0%` (無效或倒掛) | `+1.5% ~ +4.0%` (基礎對沖) | `+4.0% ~ +7.0%` (顯著分化) | `> +7.0%` (極佳多空超額) | Q5（極多）與 Q1（極空）未來前瞻收益差。必須為正數且呈現階梯單調性。 |
| **單調性評分 (Monotonicity)**| `< +0.50` (非線性/雜訊) | `+0.50 ~ +0.75` (部分單調) | `+0.75 ~ +0.95` (階梯遞增) | `+1.00` (完美階梯遞增) | Q1 到 Q5 平均收益的秩相關係數，越接近 1.0 代表評分越高收益越好。 |
| **決定係數 (R²)** | `< 0.005` (無解釋力) | `0.010 ~ 0.030` (金融典型水平) | `0.030 ~ 0.060` (強預測力) | `> 0.100` (極高，警惕過擬合) | 股票收益噪聲極大，R² 達到 1%~3% 即具備實戰邊際統計優勢。 |
| **假陽性率 (FPR)** | `> 50.0%` (噪聲過多假突破) | `35.0% ~ 50.0%` (正常震盪) | `20.0% ~ 35.0%` (信號乾淨) | `< 20.0%` (極少虛警) | 看多但實際下跌或觸發止損的比例。過高時應調高信號閾值 (|Z| ≥ σ)。 |
            """)

        # Row 2: Quantile Monotonicity Bar Chart + Cumulative Quantile Curves
        q_left, q_right = st.columns([1.05, 1.25])
        q_df = pd.DataFrame(acc_res["quantile_summary"])

        with q_left:
            bar_colors = ["#F43F5E", "#FB923C", "#94A3B8", "#38BDF8", "#10B981"]
            fig_qbar = go.Figure(
                go.Bar(
                    x=q_df["quantile"],
                    y=q_df["mean_forward_return_pct"],
                    marker_color=bar_colors,
                    text=[f"{v:+.2f}%" for v in q_df["mean_forward_return_pct"]],
                    textposition="outside",
                )
            )
            fig_qbar.update_layout(
                title=f"📊 五分位數單調性檢驗 (Q1 至 Q5 前瞻 {acc_res['forward_horizon']} 日平均收益 | 多空差 {acc_res['q5_q1_spread_pct']:+.2f}%)",
                height=390,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(11,15,25,0.72)",
                font=dict(color="#CBD5E1", size=11),
                yaxis_title=f"前瞻 {acc_res['forward_horizon']} 日平均收益率 (%)",
                margin=dict(l=15, r=15, t=45, b=15),
            )
            fig_qbar.update_xaxes(showgrid=False)
            fig_qbar.update_yaxes(showgrid=True, gridcolor="rgba(148,163,184,0.07)")
            st.plotly_chart(fig_qbar, use_container_width=True)

        with q_right:
            fig_qnav = go.Figure()
            q_palette = {
                "Q1 (極空)": "#F43F5E",
                "Q2 (偏空)": "#FB923C",
                "Q3 (中性)": "#94A3B8",
                "Q4 (偏多)": "#38BDF8",
                "Q5 (極多)": "#10B981",
            }
            for q_name, nav_s in acc_res["quantile_cum_curves"].items():
                fig_qnav.add_trace(
                    go.Scatter(
                        x=nav_s.index,
                        y=nav_s.values,
                        mode="lines",
                        name=q_name,
                        line=dict(color=q_palette.get(q_name, "#38BDF8"), width=2.4 if q_name in ("Q5 (極多)", "Q1 (極空)") else 1.4),
                    )
                )
            fig_qnav.update_layout(
                title=f"📈 分位數分組累積淨值曲線 (單調性秩相關係數 = {acc_res['monotonicity_score']:+.2f})",
                height=390,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(11,15,25,0.72)",
                font=dict(color="#CBD5E1", size=11),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                yaxis_title="累積淨值 (基準 = 1.0)",
                margin=dict(l=15, r=15, t=45, b=15),
            )
            fig_qnav.update_xaxes(showgrid=True, gridcolor="rgba(148,163,184,0.07)")
            fig_qnav.update_yaxes(showgrid=True, gridcolor="rgba(148,163,184,0.07)")
            st.plotly_chart(fig_qnav, use_container_width=True)

        # Row 3: Scatter Prediction Plot (with Regression Line & R^2) + Confusion Matrix & MAE/MFE
        sc_left, sc_right = st.columns([1.15, 1.15])
        sc_df = acc_res["scatter_df"]
        reg = acc_res["regression"]

        with sc_left:
            fig_sc = go.Figure()
            fig_sc.add_trace(
                go.Scatter(
                    x=sc_df["pred_score"],
                    y=sc_df["forward_return_pct"],
                    mode="markers",
                    name="歷史樣本點",
                    marker=dict(
                        size=6.5,
                        color=sc_df["pred_prob_pct"],
                        colorscale="Tealgrn",
                        opacity=0.72,
                        showscale=True,
                        colorbar=dict(title="上漲機率%"),
                    ),
                    text=sc_df["date"],
                    hovertemplate="日期: %{text}<br>預測得分: %{x:.1f}<br>實際前瞻漲跌: %{y:+.2f}%<extra></extra>",
                )
            )
            x_line = np.linspace(float(sc_df["pred_score"].min()), float(sc_df["pred_score"].max()), 50)
            y_line = reg["slope"] * x_line + reg["intercept"]
            fig_sc.add_trace(
                go.Scatter(
                    x=x_line,
                    y=y_line,
                    mode="lines",
                    name=f"線性回歸趨勢線 (R² = {reg['r_squared']:.3f})",
                    line=dict(color="#F59E0B", width=2.5, dash="dash"),
                )
            )
            fig_sc.update_layout(
                title=f"🎯 模型預測得分 vs 未來 {acc_res['forward_horizon']} 日實際漲跌幅散點圖 (R² = {reg['r_squared']:.3f})",
                height=395,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(11,15,25,0.72)",
                font=dict(color="#CBD5E1", size=11),
                xaxis_title="模型預測綜合得分 (-100 至 +100)",
                yaxis_title=f"未來 {acc_res['forward_horizon']} 日實際漲跌幅 (%)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=15, r=15, t=45, b=15),
            )
            fig_sc.update_xaxes(showgrid=True, gridcolor="rgba(148,163,184,0.07)")
            fig_sc.update_yaxes(showgrid=True, gridcolor="rgba(148,163,184,0.07)")
            st.plotly_chart(fig_sc, use_container_width=True)

        with sc_right:
            cm_tab1, cm_tab2 = st.tabs(["🛡️ 持倉最大有利/不利偏離 (MAE / MFE)", "🧩 訊號混淆矩陣與假陽性率 (FPR)"])
            with cm_tab1:
                fig_exc = go.Figure()
                fig_exc.add_trace(
                    go.Histogram(
                        x=sc_df["mfe_pct"],
                        name="MFE 最大有利偏離 (潛在浮盈 %)",
                        marker_color="#10B981",
                        opacity=0.70,
                    )
                )
                fig_exc.add_trace(
                    go.Histogram(
                        x=sc_df["mae_pct"],
                        name="MAE 最大不利回撤 (潛在浮虧 %)",
                        marker_color="#F43F5E",
                        opacity=0.70,
                    )
                )
                fig_exc.update_layout(
                    barmode="overlay",
                    height=330,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(11,15,25,0.72)",
                    font=dict(color="#CBD5E1", size=11),
                    xaxis_title="持倉期間最大價格偏離幅度 (%)",
                    yaxis_title="樣本頻次",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    margin=dict(l=10, r=10, t=25, b=10),
                )
                st.plotly_chart(fig_exc, use_container_width=True)

            with cm_tab2:
                cm = acc_res["confusion_matrix"]
                st.markdown(
                    f"""
                    <div class="bento-card" style="margin-top:8px;">
                        <div class="bento-title">
                            <span>🔍 多空預測混淆矩陣與停損觸發統計</span>
                            <span class="mono-num" style="color:#FBBF24;">假陽性率 FPR: {cm['fpr']*100:.1f}%</span>
                        </div>
                        <div class="exec-grid">
                            <div class="exec-cell">
                                <div class="exec-label">真陽性 (TP · 看多且獲利未停損)</div>
                                <div class="exec-val" style="color:#34D399;">{cm['tp']} 次 <span style="font-size:0.78rem; color:#94A3B8;">(精確率 {cm['precision']*100:.1f}%)</span></div>
                            </div>
                            <div class="exec-cell">
                                <div class="exec-label">假陽性 (FP · 看多但虧損或觸發停損)</div>
                                <div class="exec-val" style="color:#FB7185;">{cm['fp']} 次 <span style="font-size:0.78rem; color:#94A3B8;">(FPR {cm['fpr']*100:.1f}%)</span></div>
                            </div>
                            <div class="exec-cell">
                                <div class="exec-label">真陰性 (TN · 看空成功規避下跌)</div>
                                <div class="exec-val" style="color:#38BDF8;">{cm['tn']} 次 <span style="font-size:0.78rem; color:#94A3B8;">(停損率 {cm['stop_trigger_rate']*100:.1f}%)</span></div>
                            </div>
                            <div class="exec-cell">
                                <div class="exec-label">假陰性 (FN · 看空但標的逆勢上漲)</div>
                                <div class="exec-val" style="color:#F59E0B;">{cm['fn']} 次 <span style="font-size:0.78rem; color:#94A3B8;">(召回率 {cm['recall']*100:.1f}%)</span></div>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )