"""
web/components.py - Bento UI 卡片、表格、指標與 Plotly 圖表元件
===================================================================================
Provides GPU-safe institutional dark Bento UI CSS, horizontal factor exposure bars,
and reusable Plotly chart builders.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


def inject_terminal_and_portal_css() -> None:
    """
    Inject institutional dark Bento UI CSS + Special-Effects '香港股市預測' Portal animations.
    Ensures all pages are clean, visual, and easy to understand.
    """
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Noto+Sans+TC:wght@400;500;700;900&family=JetBrains+Mono:wght@500;600;700&display=swap');

@keyframes ambientAuroraDrift {
    0%   { transform: translate3d(-6%, -2%, 0) rotate(-3deg) scale(1.0); opacity: 0.48; }
    50%  { transform: translate3d(6%, 3%, 0) rotate(3deg) scale(1.08); opacity: 0.78; }
    100% { transform: translate3d(-6%, -2%, 0) rotate(-3deg) scale(1.0); opacity: 0.48; }
}

@keyframes specularBeamMove {
    0%   { background-position: -200% 0; }
    100% { background-position: 200% 0; }
}

@keyframes liveDotPulse {
    0%, 100% { transform: scale(1); opacity: 1; box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
    50%      { transform: scale(1.18); opacity: 0.85; box-shadow: 0 0 0 7px rgba(16, 185, 129, 0); }
}

@keyframes neonTitleShimmer {
    0%   { background-position: 0% 50%; filter: drop-shadow(0 0 18px rgba(56, 189, 248, 0.45)); }
    50%  { background-position: 100% 50%; filter: drop-shadow(0 0 32px rgba(16, 185, 129, 0.68)); }
    100% { background-position: 0% 50%; filter: drop-shadow(0 0 18px rgba(56, 189, 248, 0.45)); }
}

@keyframes cyberGridFloat {
    0%   { background-position: 0px 0px, 0px 0px; }
    100% { background-position: 0px 48px, 48px 0px; }
}

@keyframes floatUpSlow {
    0%, 100% { transform: translateY(0px); }
    50%      { transform: translateY(-8px); }
}

html, body, [class*="css"] {
    font-family: 'Inter', 'Noto Sans TC', -apple-system, BlinkMacSystemFont, sans-serif;
}

html, body, .stApp {
    background-color: #0B0F19 !important;
    color: #F8FAFC !important;
    font-family: 'Inter', 'Noto Sans TC', -apple-system, BlinkMacSystemFont, sans-serif;
}

div[data-testid='stException'], div[data-testid='stAlert'] {
    display: block !important;
}

/* Ambient Top Spotlight */
.stApp::before {
    content: "";
    position: fixed;
    top: -160px;
    left: 10%;
    width: 80vw;
    height: 340px;
    background: radial-gradient(
        ellipse at center,
        rgba(56, 189, 248, 0.15) 0%,
        rgba(99, 102, 241, 0.10) 42%,
        rgba(16, 185, 129, 0.06) 70%,
        rgba(0, 0, 0, 0) 85%
    );
    pointer-events: none;
    z-index: 0;
    animation: ambientAuroraDrift 15s ease-in-out infinite;
}

.block-container {
    padding-top: 1.4rem !important;
    padding-bottom: 2.5rem !important;
    max-width: 96% !important;
}

/* =====================================================
   Special-Effects Portal Hero ("香港股市預測")
   ===================================================== */
.hk-portal-stage {
    position: relative;
    overflow: hidden;
    border-radius: 24px;
    padding: 52px 38px 44px 38px;
    margin-bottom: 24px;
    background:
        radial-gradient(circle at 18% 22%, rgba(14, 165, 233, 0.22) 0%, transparent 45%),
        radial-gradient(circle at 82% 75%, rgba(16, 185, 129, 0.20) 0%, transparent 48%),
        radial-gradient(circle at 50% 50%, rgba(99, 102, 241, 0.14) 0%, transparent 60%),
        linear-gradient(160deg, rgba(9, 14, 28, 0.96) 0%, rgba(15, 23, 42, 0.94) 55%, rgba(6, 11, 22, 0.98) 100%);
    border: 1px solid rgba(56, 189, 248, 0.32);
    box-shadow:
        0 25px 60px -15px rgba(0, 0, 0, 0.75),
        0 0 40px -10px rgba(56, 189, 248, 0.22),
        inset 0 1px 0 rgba(255, 255, 255, 0.22);
    text-align: center;
}

.hk-portal-stage::before {
    content: "";
    position: absolute;
    inset: 0;
    background-image:
        linear-gradient(to right, rgba(56, 189, 248, 0.07) 1px, transparent 1px),
        linear-gradient(to bottom, rgba(56, 189, 248, 0.07) 1px, transparent 1px);
    background-size: 48px 48px;
    animation: cyberGridFloat 6s linear infinite;
    pointer-events: none;
}

.hk-portal-stage::after {
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 2.5px;
    background: linear-gradient(
        90deg,
        transparent 0%,
        #38BDF8 25%,
        #34D399 50%,
        #A855F7 75%,
        transparent 100%
    );
    background-size: 200% 100%;
    animation: specularBeamMove 5s linear infinite;
}

.portal-eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 10px;
    background: rgba(14, 165, 233, 0.12);
    border: 1px solid rgba(56, 189, 248, 0.38);
    border-radius: 999px;
    padding: 6px 18px;
    font-size: 0.84rem;
    font-weight: 600;
    letter-spacing: 1.5px;
    color: #7DD3FC;
    margin-bottom: 18px;
    backdrop-filter: blur(8px);
}

.portal-main-title {
    font-family: 'Noto Sans TC', 'Inter', sans-serif;
    font-size: clamp(2.8rem, 5.6vw, 4.6rem);
    font-weight: 900;
    letter-spacing: 6px;
    margin: 4px 0 14px 0;
    background: linear-gradient(
        110deg,
        #FFFFFF 0%,
        #7DD3FC 22%,
        #34D399 48%,
        #FDE047 72%,
        #38BDF8 90%,
        #FFFFFF 100%
    );
    background-size: 260% auto;
    -webkit-background-clip: text;
    color: #38BDF8;
    animation: neonTitleShimmer 7s ease-in-out infinite;
}

.portal-subtitle {
    max-width: 760px;
    margin: 0 auto 28px auto;
    font-size: 1.05rem;
    line-height: 1.65;
    color: #CBD5E1;
    font-weight: 400;
}

.portal-orb-row {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
    gap: 16px;
    margin-top: 22px;
    text-align: left;
}

.portal-stock-card {
    position: relative;
    background: linear-gradient(155deg, rgba(15, 23, 42, 0.88) 0%, rgba(11, 17, 32, 0.94) 100%);
    border: 1px solid rgba(148, 163, 184, 0.18);
    border-radius: 16px;
    padding: 18px 20px;
    backdrop-filter: blur(12px);
    transition: transform 0.3s ease, border-color 0.3s ease, box-shadow 0.3s ease;
    animation: floatUpSlow 6s ease-in-out infinite;
}

.portal-stock-card:hover {
    transform: translateY(-5px);
    border-color: rgba(56, 189, 248, 0.55);
    box-shadow: 0 16px 36px -8px rgba(14, 165, 233, 0.25);
}

/* Top Macro Status Ribbon */
.macro-ribbon {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    background: linear-gradient(180deg, rgba(15, 23, 42, 0.88) 0%, rgba(9, 13, 24, 0.92) 100%);
    border: 1px solid rgba(148, 163, 184, 0.14);
    border-top: 1px solid rgba(56, 189, 248, 0.38);
    border-radius: 12px;
    padding: 11px 18px;
    margin-bottom: 18px;
    backdrop-filter: blur(16px);
    box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
}

.macro-brand {
    display: flex;
    align-items: center;
    gap: 10px;
    font-weight: 700;
    font-size: 0.96rem;
    color: #F8FAFC;
    letter-spacing: 0.4px;
}

.live-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background-color: #10B981;
    display: inline-block;
    animation: liveDotPulse 2.4s infinite;
}

.macro-pills {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 16px;
    font-size: 0.82rem;
    color: #94A3B8;
}

.macro-item {
    display: flex;
    align-items: center;
    gap: 6px;
}

.mono-num {
    font-family: 'JetBrains Mono', monospace;
    font-variant-numeric: tabular-nums;
    font-weight: 600;
    color: #F1F5F9;
}

.badge-bull {
    background: rgba(16, 185, 129, 0.14);
    color: #34D399;
    border: 1px solid rgba(16, 185, 129, 0.35);
    padding: 2px 9px;
    border-radius: 999px;
    font-weight: 600;
    font-size: 0.76rem;
}

.badge-bear {
    background: rgba(244, 63, 94, 0.14);
    color: #FB7185;
    border: 1px solid rgba(244, 63, 94, 0.35);
    padding: 2px 9px;
    border-radius: 999px;
    font-weight: 600;
    font-size: 0.76rem;
}

.badge-warn {
    background: rgba(245, 158, 11, 0.14);
    color: #FBBF24;
    border: 1px solid rgba(245, 158, 11, 0.35);
    padding: 2px 9px;
    border-radius: 999px;
    font-weight: 600;
    font-size: 0.76rem;
}

/* Bento Grid Cards with Specular Top Light Movement */
.bento-card {
    position: relative;
    overflow: hidden;
    background: linear-gradient(165deg, rgba(17, 24, 39, 0.82) 0%, rgba(11, 15, 25, 0.92) 100%);
    border: 1px solid rgba(148, 163, 184, 0.12);
    border-radius: 14px;
    padding: 18px 20px;
    margin-bottom: 14px;
    box-shadow: 0 8px 26px -8px rgba(0, 0, 0, 0.45);
    backdrop-filter: blur(12px);
    transition: border-color 0.3s ease, box-shadow 0.3s ease;
}

.bento-card::before {
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 1.5px;
    background: linear-gradient(
        90deg,
        rgba(56, 189, 248, 0) 0%,
        rgba(56, 189, 248, 0.65) 50%,
        rgba(56, 189, 248, 0) 100%
    );
    background-size: 200% 100%;
    animation: specularBeamMove 8s linear infinite;
}

.bento-card:hover {
    border-color: rgba(56, 189, 248, 0.32);
    box-shadow: 0 12px 32px -6px rgba(14, 165, 233, 0.14);
}

.bento-title {
    font-size: 0.80rem;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: #94A3B8;
    font-weight: 600;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.exec-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 10px;
    margin-top: 12px;
}

.exec-cell {
    background: rgba(30, 41, 59, 0.42);
    border: 1px solid rgba(148, 163, 184, 0.10);
    border-radius: 10px;
    padding: 10px 12px;
}

.exec-label {
    font-size: 0.74rem;
    color: #94A3B8;
    margin-bottom: 3px;
}

.exec-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.06rem;
    font-weight: 700;
    color: #F8FAFC;
}

.factor-row {
    margin-bottom: 11px;
}

.factor-header {
    display: flex;
    justify-content: space-between;
    font-size: 0.82rem;
    margin-bottom: 4px;
    color: #CBD5E1;
}

.factor-track {
    width: 100%;
    height: 8px;
    background: rgba(30, 41, 59, 0.85);
    border-radius: 999px;
    overflow: hidden;
    position: relative;
}

.factor-fill-pos {
    height: 100%;
    background: linear-gradient(90deg, #0EA5E9, #10B981);
    border-radius: 999px;
}

.factor-fill-neg {
    height: 100%;
    background: linear-gradient(90deg, #F59E0B, #F43F5E);
    border-radius: 999px;
}

div[data-testid="stMetric"] {
    background: linear-gradient(160deg, rgba(17, 24, 39, 0.78), rgba(11, 15, 25, 0.90));
    border: 1px solid rgba(148, 163, 184, 0.13);
    border-top: 1px solid rgba(255, 255, 255, 0.18);
    border-radius: 12px;
    padding: 13px 16px;
    backdrop-filter: blur(10px);
    transition: border-color 0.25s ease, transform 0.25s ease;
}

div[data-testid="stMetric"]:hover {
    border-color: rgba(56, 189, 248, 0.45);
    transform: translateY(-2px);
}

div[data-testid="stMetricLabel"] p {
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: #94A3B8 !important;
}

div[data-testid="stMetricValue"] div {
    font-family: 'JetBrains Mono', monospace !important;
    font-variant-numeric: tabular-nums !important;
    font-size: 1.32rem !important;
    font-weight: 700 !important;
    color: #F8FAFC !important;
}

button[data-baseweb="tab"] {
    font-weight: 600 !important;
    font-size: 0.92rem !important;
    letter-spacing: 0.3px;
}
</style>
""",
        unsafe_allow_html=True,
    )


def render_factor_bars_html(
    title: str,
    subtitle: str,
    factor_dict: dict,
    max_scale: float = 3.0,
    unit: str = "σ",
) -> str:
    """
    Render sleek horizontal factor exposure bars inside a Bento card.
    Uses zero leading indentation so CommonMark never wraps inner tags into <pre><code>.
    """
    rows_list: list[str] = []
    for label, val in factor_dict.items():
        val_f = float(val)
        pct = min(100.0, max(6.0, (abs(val_f) / max_scale) * 100.0))
        fill_cls = "factor-fill-pos" if val_f >= 0 else "factor-fill-neg"
        val_color = "#34D399" if val_f > 0 else ("#FB7185" if val_f < 0 else "#94A3B8")
        dir_tag = "偏多正向" if val_f > 0.05 else ("偏空負向" if val_f < -0.05 else "中性均衡")
        row_str = (
            f'<div class="factor-row">'
            f'<div class="factor-header">'
            f'<span>{label} <small style="color:#64748B; font-size:0.74rem;">({dir_tag})</small></span>'
            f'<span class="mono-num" style="color: {val_color};">{val_f:+.2f} {unit}</span>'
            f'</div>'
            f'<div class="factor-track">'
            f'<div class="{fill_cls}" style="width: {pct:.1f}%;"></div>'
            f'</div>'
            f'</div>'
        )
        rows_list.append(row_str)

    rows_html = "".join(rows_list)
    return (
        f'<div class="bento-card">'
        f'<div class="bento-title">'
        f'<span>{title}</span>'
        f'<span style="color: #38BDF8; font-size: 0.78rem;">{subtitle}</span>'
        f'</div>'
        f'{rows_html}'
        f'</div>'
    )
