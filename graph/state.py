# Pattern: LangGraph State (강의07)
from typing import TypedDict, List


class PricingState(TypedDict):
    concert_info: dict      # artist, venue, total_seats, popularity_score, sale_start_d_day
    constraints: dict       # floor, ceiling (WTP 기반 자동 도출)
    seat_weights: dict      # {zone: {Z1, Z2, Z3, W_g}}, hedonic_betas
    wtp_model: dict         # beta_coefficients, mu_base, mu_final, sigma, beta1_over_mu, logo_mape
    pricing_result: dict    # {D60: {price, quantity, revenue, zone_prices}, ...}
    kpi: dict               # revenue_strategy1/2/3, revenue_gain_pct, mape
    insight: str            # Claude AI 전략 해설 텍스트
    report_path: str        # path to results/report.png
    current_step: str       # for console progress output
    errors: List[str]       # accumulated error messages per node
