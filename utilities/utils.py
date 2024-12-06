def format_time(t_total: float) -> str:
    return f"{t_total // 3600:.0f}H {(t_total % 3600) // 60:.0f}M {t_total % 60:.0f}S"
