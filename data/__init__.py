"""Data ingestion and Futu OpenD gateway package."""
try:
    from hk_quant_system.data.loader import (
        init_database,
        log_snapshot_to_db,
        build_suspension_mask,
        safe_download_ohlcv,
        compute_active_daily_returns,
        fetch_hk_macro_regime,
        fetch_fundamental_data,
    )
    from hk_quant_system.data.futu_gateway import (
        to_futu_ticker,
        from_futu_ticker,
        check_futu_opend_connection,
        get_futu_realtime_quote,
        get_futu_order_book,
        fetch_realtime_spot_price,
    )
except (ImportError, ValueError):
    from .loader import (
        init_database,
        log_snapshot_to_db,
        build_suspension_mask,
        safe_download_ohlcv,
        compute_active_daily_returns,
        fetch_hk_macro_regime,
        fetch_fundamental_data,
    )
    from .futu_gateway import (
        to_futu_ticker,
        from_futu_ticker,
        check_futu_opend_connection,
        get_futu_realtime_quote,
        get_futu_order_book,
        fetch_realtime_spot_price,
    )
