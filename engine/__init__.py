"""Engine package: Alpha, Risk, and Backtesting."""
try:
    from hk_quant_system.engine.alpha import (
        compute_wilder_rsi,
        compute_intraday_vwap_and_tod_volume,
        compute_factor_exposure_and_ic_weights,
        generate_day_trading_signal,
        analyze_stock_trend,
    )
    from hk_quant_system.engine.risk import (
        compute_ledoit_wolf_cov,
        compute_rmt_denoised_cov,
        optimize_portfolio_weights,
        compute_deflated_sharpe_ratio,
        generate_execution_orders,
        analyze_portfolio_health,
    )
    from hk_quant_system.engine.backtest import (
        run_strategy_backtest,
        evaluate_historical_prediction_accuracy,
    )
except (ImportError, ValueError):
    from .alpha import (
        compute_wilder_rsi,
        compute_intraday_vwap_and_tod_volume,
        compute_factor_exposure_and_ic_weights,
        generate_day_trading_signal,
        analyze_stock_trend,
    )
    from .risk import (
        compute_ledoit_wolf_cov,
        compute_rmt_denoised_cov,
        optimize_portfolio_weights,
        compute_deflated_sharpe_ratio,
        generate_execution_orders,
        analyze_portfolio_health,
    )
    from .backtest import (
        run_strategy_backtest,
        evaluate_historical_prediction_accuracy,
    )
