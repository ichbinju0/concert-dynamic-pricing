import pytest
from tools.simulation import (
    simulate_strategy1, simulate_strategy2, simulate_strategy3,
    calculate_kpi, find_optimal_static_price,
)

_WTP = {"mu_final": 200000, "sigma": 30000, "beta1_over_mu": -0.0025}
_LP  = {
    "D60": {"price": 180000.0, "quantity": 3500.0, "revenue": 6.3e8},
    "D30": {"price": 170000.0, "quantity": 3200.0, "revenue": 5.44e8},
    "D14": {"price": 165000.0, "quantity": 3100.0, "revenue": 5.115e8},
    "D7":  {"price": 160000.0, "quantity": 3000.0, "revenue": 4.8e8},
    "D1":  {"price": 155000.0, "quantity": 2900.0, "revenue": 4.495e8},
}


def test_strategy1_fixed_price():
    result = simulate_strategy1(154000, _WTP, 15000)
    assert set(result.keys()) == {"D60", "D30", "D14", "D7", "D1"}
    for v in result.values():
        assert v["price"] == 154000


def test_strategy3_passthrough():
    assert simulate_strategy3(_LP) == _LP


def test_kpi_keys():
    s1 = simulate_strategy1(154000, _WTP, 15000)
    s2 = simulate_strategy2(find_optimal_static_price(130900, 308000, _WTP, 15000), _WTP, 15000)
    s3 = simulate_strategy3(_LP)
    kpi = calculate_kpi(s1, s2, s3, 154000, 15000)
    for k in ["revenue_gain_pct", "cpk", "mape"]:
        assert k in kpi


def test_cpk_is_float():
    s1 = simulate_strategy1(154000, _WTP, 15000)
    s3 = simulate_strategy3(_LP)
    kpi = calculate_kpi(s1, s1, s3, 154000, 15000)
    assert isinstance(kpi["cpk"], float)
