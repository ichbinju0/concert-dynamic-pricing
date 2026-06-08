# Pattern: Tool Node (강의05)
import numpy as np
from scipy.stats import norm

_INTERVALS = ["D60", "D30", "D14", "D7", "D1"]
_D_DAYS    = {"D60": 60, "D30": 30, "D14": 14, "D7": 7, "D1": 1}


def _d_factor(beta1_over_mu: float, d_day: int) -> float:
    return max(0.5, min(1.0 + beta1_over_mu * (14 - d_day), 1.5))


def _demand(price: float, mu_adj: float, sigma: float, total_seats: int) -> float:
    return float(total_seats * (1 - norm.cdf(price, mu_adj, sigma)))


def simulate_strategy1(base_price: int, wtp_model: dict, total_seats: int) -> dict:
    """Strategy 1: Fixed pricing at mu_final (WTP 회귀 평균값) for all intervals.

    official_price 미사용 — 티켓베이 데이터 기반 mu_final이 기준가.
    """
    b1m   = wtp_model["beta1_over_mu"]
    mu    = wtp_model["mu_final"]
    sigma = wtp_model["sigma"]
    result = {}
    for interval in _INTERVALS:
        mu_adj = mu * _d_factor(b1m, _D_DAYS[interval])
        qt = _demand(base_price, mu_adj, sigma, total_seats)
        result[interval] = {
            "price":    float(base_price),
            "quantity": qt,
            "revenue":  float(base_price) * qt,
        }
    return result


def find_optimal_static_price(
    floor_price: int, ceiling_price: int, wtp_model: dict, total_seats: int
) -> float:
    """Find single price maximising total revenue across all intervals."""
    b1m   = wtp_model["beta1_over_mu"]
    mu    = wtp_model["mu_final"]
    sigma = wtp_model["sigma"]
    best_price, best_rev = float(floor_price), 0.0
    for p in np.linspace(floor_price, ceiling_price, 50):
        total_rev = sum(
            p * _demand(p, mu * _d_factor(b1m, _D_DAYS[iv]), sigma, total_seats)
            for iv in _INTERVALS
        )
        if total_rev > best_rev:
            best_rev, best_price = total_rev, float(p)
    return best_price


def simulate_strategy2(static_price: float, wtp_model: dict, total_seats: int) -> dict:
    """Strategy 2: Optimal static price applied to all intervals."""
    b1m   = wtp_model["beta1_over_mu"]
    mu    = wtp_model["mu_final"]
    sigma = wtp_model["sigma"]
    result = {}
    for interval in _INTERVALS:
        mu_adj = mu * _d_factor(b1m, _D_DAYS[interval])
        qt = _demand(static_price, mu_adj, sigma, total_seats)
        result[interval] = {
            "price":    static_price,
            "quantity": qt,
            "revenue":  static_price * qt,
        }
    return result


def simulate_strategy3(lp_prices: dict) -> dict:
    """Strategy 3: Dynamic pricing — LP output passed through directly."""
    return lp_prices


def calculate_kpi(
    strategy1: dict,
    strategy2: dict,
    strategy3: dict,
    mu_final: float,
    total_seats: int,
) -> dict:
    """Compute Revenue Gain and MAPE KPIs.

    MAPE = LP 동적가격(S3)이 mu_final(WTP 기준가) 대비 얼마나 벗어나는지.
    official_price 미사용 — 티켓베이 회귀 기반 mu_final이 기준.
    """
    rev1 = sum(v["revenue"] for v in strategy1.values())
    rev2 = sum(v["revenue"] for v in strategy2.values())
    rev3 = sum(v["revenue"] for v in strategy3.values())

    revenue_gain = (rev3 - rev1) / rev1 * 100 if rev1 > 0 else 0.0

    # MAPE: LP 동적가격이 mu_final 대비 얼마나 달라지는지
    s3_prices = np.array([v["price"] for v in strategy3.values()])
    mape = float(np.mean(np.abs(s3_prices - mu_final) / mu_final)) if mu_final else 0.0

    return {
        "revenue_strategy1": rev1,
        "revenue_strategy2": rev2,
        "revenue_strategy3": rev3,
        "revenue_gain_pct":  revenue_gain,
        "mape":              mape,
    }


# ── Price Sensitivity Scenario Analysis ───────────────────────────────────────
#
# 이탈 기준: 가격이 μ_final 대비 10% 오를 때마다 churn_per_10pct 만큼 수요 추가 이탈
# 브랜드 패널티: 이탈한 팬 1명 × μ_final × brand_penalty = 미래 LTV 손실 추정
# 가격 탐색 범위: floor ~ 3×μ_final (ceiling 제거 → 이탈이 자연 천장 역할)
#
# S1: 이탈 0% (현재 WTP 모델 그대로 — baseline)
# S2: 10% 인상당 3% 이탈          (소비자행동론 엔터테인먼트 탄력성 하한, ~-0.3)
# S3: 10% 인상당 6% 이탈          (K-pop 팬덤 중간 추정, ~-0.6)
# S4: 10% 인상당 10% + 브랜드     (단위탄력성 구간, SNS 불만 시작)
# S5: 10% 인상당 15% + 브랜드     (탈덕·안티 전환, 브랜드 훼손 심각)

_SENSITIVITY_SCENARIOS = [
    {
        "name":            "S1 (Current Model)",
        "label":           "No churn  |  No penalty",
        "churn_per_10pct": 0.00,
        "brand_penalty":   0.00,
        "color":           "#2ecc71",
    },
    {
        "name":            "S2",
        "label":           "+10% price → 3% churn",
        "churn_per_10pct": 0.03,
        "brand_penalty":   0.00,
        "color":           "#3498db",
    },
    {
        "name":            "S3",
        "label":           "+10% price → 6% churn",
        "churn_per_10pct": 0.06,
        "brand_penalty":   0.00,
        "color":           "#f39c12",
    },
    {
        "name":            "S4  (+ Brand Penalty)",
        "label":           "+10% price → 10% churn  +  brand damage",
        "churn_per_10pct": 0.10,
        "brand_penalty":   0.05,
        "color":           "#e67e22",
    },
    {
        "name":            "S5  (+ Brand Penalty)",
        "label":           "+10% price → 15% churn  +  brand damage",
        "churn_per_10pct": 0.15,
        "brand_penalty":   0.12,
        "color":           "#e74c3c",
    },
]


def simulate_sensitivity_scenarios(
    wtp_model: dict,
    total_seats: int,
    floor_price: float,
    d_day: int = 14,
) -> list[dict]:
    """5가지 가격 민감도 시나리오별 순수익 곡선을 계산합니다 (D-day 고정).

    이탈 공식:
      price_increase_pct = (P − floor_price) / floor_price   (floor 대비 인상률)
      churn_frac         = churn_per_10pct × (price_increase_pct / 0.10)
      net_demand         = WTP_demand × max(0, 1 − churn_frac)
      brand_cost         = churned_fans × μ_final × brand_penalty
      net_revenue        = P × net_demand − brand_cost

    가격 탐색 범위: floor_price ~ 3×μ_final (ceiling 없음 — 이탈이 자연 천장)

    Returns:
        list of dicts with keys:
          name, label, color,
          price_candidates, net_revenues,
          optimal_price, optimal_net_revenue, revenue_gain_vs_s1_pct
    """
    mu     = wtp_model["mu_final"]
    sigma  = wtp_model["sigma"]
    b1m    = wtp_model["beta1_over_mu"]
    mu_adj = mu * _d_factor(b1m, d_day)

    # Ceiling 없이 3×μ_final 까지 탐색
    price_max        = max(mu * 3.0, floor_price * 2.0)
    price_candidates = np.linspace(floor_price, price_max, 80)

    all_results = []
    s1_opt_rev  = None  # Revenue Gain 기준 (S1 최적수익)

    for sc in _SENSITIVITY_SCENARIOS:
        net_revs = []

        for p in price_candidates:
            base_q   = _demand(p, mu_adj, sigma, total_seats)

            # 이탈: floor_price 대비 가격인상률 기준
            # "floor 대비 10% 오를 때마다 churn_per_10pct 만큼 추가 이탈"
            pct_above_floor = max(0.0, (p - floor_price) / floor_price)
            churn_frac      = min(0.99, sc["churn_per_10pct"] * (pct_above_floor / 0.10))

            net_q      = base_q * (1.0 - churn_frac)
            churned    = base_q - net_q
            brand_cost = churned * mu * sc["brand_penalty"]
            net_rev    = p * net_q - brand_cost

            net_revs.append(float(net_rev))

        opt_idx     = int(np.argmax(net_revs))
        opt_price   = float(price_candidates[opt_idx])
        opt_rev     = float(net_revs[opt_idx])

        if s1_opt_rev is None:
            s1_opt_rev = opt_rev  # S1이 baseline

        all_results.append({
            "name":                   sc["name"],
            "label":                  sc["label"],
            "color":                  sc["color"],
            "churn_per_10pct":        sc["churn_per_10pct"],
            "brand_penalty":          sc["brand_penalty"],
            "price_candidates":       price_candidates.tolist(),
            "net_revenues":           net_revs,
            "optimal_price":          opt_price,
            "optimal_net_revenue":    opt_rev,
            "revenue_gain_vs_s1_pct": 0.0,   # S1 기준 재계산 후 채워짐
        })

    # revenue_gain_vs_s1_pct: S1 최적수익 기준으로 통일 재계산
    s1_opt = all_results[0]["optimal_net_revenue"]
    for r in all_results:
        r["revenue_gain_vs_s1_pct"] = round(
            (r["optimal_net_revenue"] - s1_opt) / s1_opt * 100 if s1_opt else 0.0, 2
        )

    return all_results
