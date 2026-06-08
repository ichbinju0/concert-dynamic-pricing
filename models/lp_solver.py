import numpy as np
from scipy.stats import norm
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, PULP_CBC_CMD

from tools.file_manager import save_json

_INTERVALS          = [("D60", 60), ("D30", 30), ("D14", 14), ("D7", 7), ("D1", 1)]
_N_PRICE_CANDIDATES = 20


def compute_d_factor(beta1_over_mu: float, d_day: int) -> float:
    return max(0.5, min(1.0 + beta1_over_mu * (14 - d_day), 1.5))


def solve_lp_for_interval(
    interval: str,
    d_day: int,
    mu_final: float,
    sigma: float,
    total_seats: int,
    floor_price: int,
    ceiling_price: int,
    beta1_over_mu: float,
) -> dict:
    mu_adj           = mu_final * compute_d_factor(beta1_over_mu, d_day)
    price_candidates = np.linspace(floor_price, ceiling_price, _N_PRICE_CANDIDATES)
    qt_lookup        = {p: float(total_seats * (1 - norm.cdf(p, mu_adj, sigma))) for p in price_candidates}

    prob = LpProblem(f"pricing_{interval}", LpMaximize)
    x    = {p: LpVariable(f"x_{p:.0f}", cat="Binary") for p in price_candidates}

    prob += lpSum(p * qt_lookup[p] * x[p] for p in price_candidates)
    prob += lpSum(x[p] for p in price_candidates) == 1
    prob.solve(PULP_CBC_CMD(msg=0))

    chosen       = [p for p in price_candidates if (x[p].value() or 0) > 0.5]
    chosen_price = float(chosen[0]) if chosen else float(price_candidates[_N_PRICE_CANDIDATES // 2])
    return {
        "price":    chosen_price,
        "quantity": qt_lookup[chosen_price],
        "revenue":  chosen_price * qt_lookup[chosen_price],
    }


def compute_zone_prices(base_price: float, zones: dict, floor: int, ceiling: int) -> dict:
    wg_values = [v["W_g"] for v in zones.values()]
    wg_mean   = max(float(np.mean(wg_values)), 1e-9) if wg_values else 1.0
    return {
        zone: int(np.clip(base_price * (v["W_g"] / wg_mean), floor, ceiling))
        for zone, v in zones.items()
    }


def run_lp_agent(state: dict) -> dict:
    print("[lp] Running LP optimization...")
    ci     = state["concert_info"]
    wtp    = state["wtp_model"]
    cons   = state["constraints"]
    zones  = state.get("seat_weights", {}).get("zones", {})
    errors = list(state.get("errors", []))

    results = {}
    for interval, d_day in _INTERVALS:
        base = solve_lp_for_interval(
            interval      = interval,
            d_day         = d_day,
            mu_final      = wtp["mu_final"],
            sigma         = wtp["sigma"],
            total_seats   = ci["total_seats"],
            floor_price   = cons["floor"],
            ceiling_price = cons["ceiling"],
            beta1_over_mu = wtp["beta1_over_mu"],
        )
        base["zone_prices"] = compute_zone_prices(base["price"], zones, cons["floor"], cons["ceiling"])
        results[interval]   = base

    print("[lp] Optimal prices computed for all D-day intervals")
    save_json("lp_prices.json", results)
    return {"pricing_result": results, "errors": errors}