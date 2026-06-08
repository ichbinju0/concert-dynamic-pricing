import pytest
from agents.lp_agent import compute_d_factor, solve_lp_for_interval, run_lp_agent


def test_d14_is_one():
    assert abs(compute_d_factor(-0.0025, 14) - 1.0) < 1e-9


def test_clamp_lower():
    # beta=+999, d_day=60: 1 + 999*(14-60) = very negative → clamped to 0.5
    assert compute_d_factor(999.0, 60) == 0.5


def test_clamp_upper():
    # beta=-999, d_day=60: 1 + (-999)*(14-60) = very positive → clamped to 1.5
    assert compute_d_factor(-999.0, 60) == 1.5


def test_lp_valid_result():
    result = solve_lp_for_interval(
        interval="D14", d_day=14,
        mu_final=200000, sigma=30000,
        total_seats=15000, floor_price=130000, ceiling_price=310000,
        beta1_over_mu=-0.0025,
    )
    assert 130000 <= result["price"] <= 310000
    assert result["quantity"] >= 0


def test_run_lp_agent_5_intervals():
    state = {
        "concert_info": {"total_seats": 15000, "official_price": 154000,
                         "popularity_score": 6, "sale_start_d_day": 60},
        "constraints": {"floor": 130900, "ceiling": 308000},
        "wtp_model": {"mu_final": 200000, "sigma": 30000,
                      "beta1_over_mu": -0.0025, "logo_mape": 0.14},
        "errors": [],
    }
    update = run_lp_agent(state)
    assert set(update["pricing_result"].keys()) == {"D60", "D30", "D14", "D7", "D1"}
