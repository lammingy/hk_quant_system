#!/usr/bin/env python3
"""
main.py - CLI 命令列執行入口
===================================================================================
Usage Examples:
  python main.py --web                            # 啟動 Streamlit 視覺化終端儀表板 (自動探測可用端口)
  python main.py --analyze 0700.HK                # 執行雙週期多因子趨勢機率與 VIF 共線性診斷
  python main.py --signal 0700.HK                 # 生成無未來函數日內量化訊號與港交所微觀規費試算
  python main.py --validate 0700.HK --horizon 21  # 執行多週期歷史勝率與五分位數單調性檢驗
  python main.py --backtest                       # 執行投資組合滾動走步前瞻樣本外回測與 DSR 檢驗
"""

from __future__ import annotations

import argparse
import socket
import subprocess
import sys
from pathlib import Path

_CURRENT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _CURRENT_DIR.parent

for p in [str(_CURRENT_DIR), str(_REPO_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)


def find_available_port(preferred_port: int = 8501, max_search: int = 50) -> int:
    """Find preferred_port or the next available TCP port on localhost."""
    for p in range(preferred_port, preferred_port + max_search):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", p))
                return p
        except OSError:
            continue
    return preferred_port


def run_web_dashboard(port: int | None = None) -> None:
    """Launch Streamlit web terminal with intelligent auto-port discovery."""
    app_path = _CURRENT_DIR / "web" / "app.py"
    if not app_path.exists():
        app_path = _CURRENT_DIR / "app.py"

    target_port = port if (port is not None and port > 0) else 8501
    active_port = find_available_port(target_port)

    if active_port != target_port:
        print(f"⚠️ 偵測到端口 {target_port} 正被其他處理程序佔用，已自動平滑切換至可用端口 {active_port}！")

    print(f"🚀 啟動香港股市預測 | HKEX AlphaEngine 機構級量化終端 (Port: {active_port})...")
    print(f"🌐 本地瀏覽器訪問網址: http://localhost:{active_port}")

    cmd = [
        sys.executable, "-m", "streamlit", "run", str(app_path),
        "--server.port", str(active_port),
        "--server.headless", "false",
    ]
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n🛑 終端已安全關閉。")


def run_trend_analysis(ticker: str) -> None:
    """Run Swing Trend Multi-Factor Evaluation & VIF Diagnostics."""
    try:
        from hk_quant_system.config.settings import normalize_hk_ticker
        from hk_quant_system.engine.alpha import analyze_stock_trend
    except (ImportError, ValueError):
        from config.settings import normalize_hk_ticker
        from engine.alpha import analyze_stock_trend

    norm_t = normalize_hk_ticker(ticker)
    print(f"\n{'='*78}")
    print(f"📊 HKEX AlphaEngine 多因子波段趨勢診斷: {norm_t}")
    print(f"{'='*78}")

    trend_res, err = analyze_stock_trend(norm_t)
    if err or not trend_res:
        print(f"❌ 診斷失敗: {err}")
        return

    print(f"• 標的代號: {trend_res['ticker']}")
    print(f"• 最新實盤現價: HK$ {trend_res['price']:.2f} (決策基準價 Close[t-1]: HK$ {trend_res['decision_price']:.2f})")
    print(f"• 次日開盤預估撮合價: HK$ {trend_res['execution_price']:.2f}")
    print(f"• 趨勢上漲機率 P(Up): {trend_res['up_prob']*100:.1f}% (合成因子暴露 Z = {trend_res['composite_zscore']:+.2f}σ)")
    print(f"• 短期戰術上漲機率: {trend_res['short_term_prob']*100:.1f}% | 長期戰略上漲機率: {trend_res['long_term_prob']*100:.1f}%")
    print(f"• 綜合趨勢狀態: {trend_res['trend']}")

    print(f"\n[四大核心因子標準化暴露與 VIF 共線性診斷]")
    for row in trend_res.get("factor_exposure_matrix", []):
        print(
            f"  • {row['factor_name']:<28}: Z = {row['z_score']:+.2f}σ | "
            f"Rank-IC = {row['rank_ic']:+.3f} | 權重 = {row['ic_weight_pct']:4.1f}% | "
            f"VIF = {row['vif']:.2f} ({row['collinearity_status']})"
        )

    print(f"\n[量化診斷明細]")
    for r in trend_res.get("reasons", []):
        print(f"  • {r}")
    print(f"{'='*78}\n")


def run_intraday_signal(ticker: str, interval: str = "15m") -> None:
    """Generate Intraday Execution Signal & Tick-Adjusted Stops/Targets."""
    try:
        from hk_quant_system.config.settings import load_portfolio_config, normalize_hk_ticker
        from hk_quant_system.engine.alpha import generate_day_trading_signal
    except (ImportError, ValueError):
        from config.settings import load_portfolio_config, normalize_hk_ticker
        from engine.alpha import generate_day_trading_signal

    cfg = load_portfolio_config()
    norm_t = normalize_hk_ticker(ticker)
    print(f"\n{'='*78}")
    print(f"⚡ HKEX AlphaEngine 日內高頻執行訊號: {norm_t} ({interval})")
    print(f"{'='*78}")

    sig_res, err = generate_day_trading_signal(
        norm_t,
        min_adv_hkd=float(cfg.get("min_adv_hkd", 5_000_000.0)),
        interval=interval,
        broker_comm_rate=float(cfg.get("broker_comm_rate", 0.0003)),
        min_broker_comm_hkd=float(cfg.get("min_broker_comm_hkd", 15.0)),
        almgren_gamma=float(cfg.get("almgren_gamma", 0.35)),
    )
    if err or not sig_res:
        print(f"❌ 訊號生成失敗: {err}")
        return

    mc = sig_res["microstructure"]
    print(f"• 標的代號: {sig_res['ticker']} (每手 {sig_res['lot_size']} 股 | 法定 Tick ${sig_res['tick_size']})")
    print(f"• 交易決策指令: {sig_res['action']}")
    print(f"• 上漲趨勢機率 P(Up): {sig_res['up_prob']*100:.1f}% (合成 Z = {sig_res['composite_zscore']:+.2f}σ)")
    print(f"• 決策基準價 (Close[t-1]): HK$ {sig_res['decision_price']:.2f}")
    print(f"• 實盤撮合價 (Open[t] + 滑價): HK$ {sig_res['execution_price']:.2f} (衝擊滑價 {mc['slippage_bps']:.1f} bps)")
    print(f"• 每日 09:30 重置之 VWAP: HK$ {sig_res['vwap']:.2f} | Wilder RSI(14): {sig_res['rsi']:.1f}")
    print(f"• 硬性止損價 (Tick對齊): HK$ {sig_res['stop_loss_price']:.2f}")
    print(f"• 目標止盈價 (Tick對齊): HK$ {sig_res['take_profit_price']:.2f}")
    print(f"• 單筆法定規費合計: HK$ {mc['total_statutory_fees_hkd']:.2f} (含印花稅 HK$ {mc['stamp_duty_hkd']:.0f})")
    print(f"{'='*78}\n")


def run_accuracy_validation(ticker: str, horizon: int = 21) -> None:
    """Run Multi-Horizon Historical Accuracy & Monotonicity Validation."""
    try:
        from hk_quant_system.config.settings import normalize_hk_ticker
        from hk_quant_system.engine.backtest import evaluate_historical_prediction_accuracy
    except (ImportError, ValueError):
        from config.settings import normalize_hk_ticker
        from engine.backtest import evaluate_historical_prediction_accuracy

    norm_t = normalize_hk_ticker(ticker)
    print(f"\n{'='*78}")
    print(f"🔬 HKEX AlphaEngine 歷史預測精準度與單調性檢驗: {norm_t} (前瞻 {horizon} 日)")
    print(f"{'='*78}")

    acc_res, err = evaluate_historical_prediction_accuracy(norm_t, forward_horizon=horizon)
    if err or not acc_res:
        print(f"❌ 檢驗失敗: {err}")
        return

    print(f"• 標的代號: {acc_res['ticker']} | 前瞻評估週期: {acc_res['forward_horizon']} 個交易日")
    print(f"• 有效觸發訊號數: {acc_res['sample_size']} 次 (總評估日數: {acc_res['total_bars_evaluated']} 日)")
    print(f"• 歷史方向勝率 (Hit Rate): {acc_res['hit_rate']*100:.1f}%")
    print(f"• 平均 Rank-IC: {acc_res['mean_ic']:+.3f} | 資訊比率 IC-IR: {acc_res['ic_ir']:+.2f}")
    print(f"• 五分位數單調性秩相關: {acc_res['monotonicity_score']:+.2f} | Q5-Q1 多空收益差: {acc_res['q5_q1_spread_pct']:+.2f}%")
    print(f"• 獲利因子 (Profit Factor): {acc_res['profit_factor']:.2f} | 盈虧比: {acc_res['win_loss_ratio']:.2f}x")
    print(f"{'='*78}\n")


def run_portfolio_backtest() -> None:
    """Run Portfolio Optimization & Walk-Forward OOS Backtest."""
    try:
        from hk_quant_system.config.settings import load_portfolio_config
        from hk_quant_system.engine.risk import analyze_portfolio_health
    except (ImportError, ValueError):
        from config.settings import load_portfolio_config
        from engine.risk import analyze_portfolio_health

    cfg = load_portfolio_config()
    print(f"\n{'='*78}")
    print(f"🚀 HKEX AlphaEngine 穩健組合最佳化與走步前瞻回測")
    print(f"{'='*78}")

    res = analyze_portfolio_health(
        cfg.get("holdings", []),
        total_capital=float(cfg.get("total_capital_hkd", 300_000.0)),
        rf_rate=float(cfg.get("risk_free_rate", 0.04)),
        max_single_weight=float(cfg.get("max_single_position_pct", 0.35)),
        min_adv_hkd=float(cfg.get("min_adv_hkd", 5_000_000.0)),
        alloc_method=str(cfg.get("alloc_method", "hrp")),
        num_trials=int(cfg.get("num_backtest_trials", 5)),
        cov_estimator=str(cfg.get("cov_estimator", "ledoit_wolf")),
        max_turnover=float(cfg.get("max_turnover", 0.40)),
    )
    if not res:
        print("❌ 組合回測失敗，請檢查持倉代號是否具備至少 120 個有效交易日。")
        return

    bt = res.get("backtest")
    print(f"• 組合健康度評分: {res['health_score']}/100 ({res['health_status']})")
    print(f"• 現有持倉總市值: HK$ {res['total_value']:,.0f}")
    if bt:
        print(f"• 樣本外幾何年化報酬 (CAGR): {bt['cagr']*100:+.2f}% | 總回報: {bt['total_return']*100:+.1f}%")
        print(f"• 樣本外夏普比率 (Sharpe): {bt['sharpe']:.2f} | 年化波動率: {bt['ann_volatility']*100:.1f}%")
        print(f"• 最大回撤 (Max Drawdown): {bt['max_drawdown']*100:.1f}% | 單日 CVaR(95%): {res['cvar_95']*100:.2f}%")
        print(f"• DSR 抗過擬合信心度: {bt['dsr_prob']*100:.1f}% (p = {bt['dsr_pvalue']:.3f})")
    print(f"{'='*78}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="香港股市預測 | HKEX AlphaEngine 機構級量化終端 (CLI)")
    parser.add_argument("--web", action="store_true", help="啟動 Streamlit 視覺化終端儀表板")
    parser.add_argument("--port", type=int, default=8501, help="Streamlit 監聽端口 (預設: 8501)")
    parser.add_argument("--analyze", type=str, metavar="TICKER", help="執行個股雙週期多因子趨勢評估與 VIF 診斷 (如 0700.HK)")
    parser.add_argument("--signal", type=str, metavar="TICKER", help="生成日內高頻量化訊號與微觀結構摩擦成本 (如 0700.HK)")
    parser.add_argument("--validate", type=str, metavar="TICKER", help="執行多週期歷史預測精準度與五分位數單調性檢驗 (如 0700.HK)")
    parser.add_argument("--horizon", type=int, default=21, help="歷史精準度檢驗之前瞻交易日 (預設: 21)")
    parser.add_argument("--backtest", action="store_true", help="執行投資組合滾動走步前瞻樣本外回測與 DSR 檢驗")

    args = parser.parse_args()
    if args.web:
        run_web_dashboard(port=args.port)
    elif args.analyze:
        run_trend_analysis(args.analyze)
    elif args.signal:
        run_intraday_signal(args.signal)
    elif args.validate:
        run_accuracy_validation(args.validate, horizon=args.horizon)
    elif args.backtest:
        run_portfolio_backtest()
    else:
        print("\n" + "="*78)
        print("🏛️  香港股市預測 | HKEX AlphaEngine 機構級量化終端 (Modular Edition)")
        print("="*78)
        print("未指定命令參數，可使用以下指令：")
        print("  python main.py --web                     # 啟動 Streamlit 互動式終端")
        print("  python main.py --analyze 0700.HK         # 個股多因子與 VIF 診斷")
        print("  python main.py --signal 0700.HK          # 日內執行訊號與止損止盈")
        print("  python main.py --validate 0700.HK        # 歷史預測精準度與單調性")
        print("  python main.py --backtest                # 投資組合走步前瞻回測與 DSR")
        print("  python main.py --help                    # 檢視完整參數指南\n")


if __name__ == "__main__":
    main()
