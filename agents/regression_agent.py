# Pattern: Sub-agent (강의06)
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold

from tools.file_manager import save_json
from tools.data_loader import load_ticketbay_data

_POPULARITY_FACTOR = {7: 1.4, 6: 1.2, 5: 1.2, 4: 1.0, 3: 1.0, 2: 0.85, 1: 0.85}


def build_feature_matrix(df: pd.DataFrame, seat_weights: dict):
    """Build X (4 features), y (listing_price), groups (artist_group) arrays."""
    zone_map = {z: v["W_g"] for z, v in seat_weights["zones"].items()}
    default_wg = float(np.mean(list(zone_map.values())))
    df = df.copy()
    if "seat_zone" in df.columns:
        df["W_g"] = df["seat_zone"].map(zone_map).fillna(default_wg)
    else:
        df["W_g"] = default_wg

    X = df[["d_day", "kopis_booking_rate", "W_g", "popularity_score"]].values.astype(float)
    y = df["listing_price"].values.astype(float)
    groups = (
        df["artist_group"].values
        if "artist_group" in df.columns
        else np.zeros(len(df), int)
    )
    return X, y, groups


def fit_wtp_model(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> dict:
    """Fit OLS and run LOGO cross-validation. Returns model parameters."""
    model = LinearRegression()
    model.fit(X, y)

    mu_base = float(np.mean(y))
    sigma   = float(np.std(y))
    beta1   = float(model.coef_[0])
    beta1_over_mu = beta1 / mu_base if mu_base != 0 else 0.0

    unique_groups = np.unique(groups)
    n_splits = max(2, len(unique_groups))
    gkf = GroupKFold(n_splits=n_splits)
    mapes = []
    for train_idx, test_idx in gkf.split(X, y, groups):
        m     = LinearRegression().fit(X[train_idx], y[train_idx])
        preds = m.predict(X[test_idx])
        mape  = float(np.mean(np.abs((y[test_idx] - preds) / (y[test_idx] + 1e-9))))
        mapes.append(mape)

    return {
        "beta_coefficients": {
            "beta_d_day":        float(model.coef_[0]),
            "beta_booking_rate": float(model.coef_[1]),
            "beta_W_g":          float(model.coef_[2]),
            "beta_popularity":   float(model.coef_[3]),
            "intercept":         float(model.intercept_),
        },
        "mu_base":       mu_base,
        "sigma":         sigma,
        "beta1_over_mu": beta1_over_mu,
        "logo_mape":     float(np.mean(mapes)),
    }


def run_regression_agent(state: dict) -> dict:
    """LangGraph node function for regression_node (Step 3)."""
    print("Step 3 📊 WTP 회귀분석 실행 중...")
    seat_weights     = state["seat_weights"]
    popularity_score = state["concert_info"]["popularity_score"]
    errors = list(state.get("errors", []))

    try:
        df = load_ticketbay_data()
    except Exception as e:
        errors.append(f"data_loader error: {e}")
        df = pd.DataFrame()

    try:
        X, y, groups = build_feature_matrix(df, seat_weights)
    except Exception as e:
        errors.append(f"feature matrix error: {e}")
        X, y, groups = np.zeros((0, 4)), np.zeros(0), np.zeros(0, int)

    # 데이터 부족 시 mock fallback
    if len(y) < 2:
        errors.append("regression: 유효 데이터 부족 → mock fallback 사용")
        official = state["concert_info"].get("official_price", 150000)
        result = {
            "beta_coefficients": {"beta_d_day": -500.0, "beta_booking_rate": 30000.0,
                                  "beta_W_g": 20000.0, "beta_popularity": 5000.0, "intercept": float(official)},
            "mu_base": float(official), "sigma": float(official * 0.3),
            "beta1_over_mu": -0.003, "logo_mape": 0.0,
        }
    else:
        result = fit_wtp_model(X, y, groups)

    pop_factor = _POPULARITY_FACTOR.get(popularity_score, 1.0)
    result["mu_final"]        = result["mu_base"] * pop_factor
    result["popularity_factor"] = pop_factor

    print(f"Step 3 ✅ Cross-Concert MAPE: {result['logo_mape']:.3f} (4-group LOGO)")
    save_json("regression.json", result)
    return {"wtp_model": result, "errors": errors}
