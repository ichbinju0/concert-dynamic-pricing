import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold

from tools.file_manager import save_json
from tools.data_loader import load_ticketbay_data

_POPULARITY_FACTOR = {7: 1.4, 6: 1.2, 5: 1.2, 4: 1.0, 3: 1.0, 2: 0.85, 1: 0.85}


def build_feature_matrix(df: pd.DataFrame, seat_weights: dict):
    zone_map   = {z: v["W_g"] for z, v in seat_weights["zones"].items()}
    default_wg = float(np.mean(list(zone_map.values())))
    df = df.copy()
    df["W_g"] = df["seat_zone"].map(zone_map).fillna(default_wg) if "seat_zone" in df.columns else default_wg

    X      = df[["d_day", "W_g", "popularity_score"]].values.astype(float)
    y      = df["listing_price"].values.astype(float)
    groups = df["artist_group"].values if "artist_group" in df.columns else np.zeros(len(df), int)
    return X, y, groups


def fit_wtp_model(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> dict:
    model = LinearRegression().fit(X, y)

    mu_base       = float(np.mean(y))
    sigma         = float(np.std(y))
    beta1_over_mu = float(model.coef_[0]) / mu_base if mu_base != 0 else 0.0

    n_splits = max(2, len(np.unique(groups)))
    mapes = [
        float(np.mean(np.abs((y[test] - LinearRegression().fit(X[train], y[train]).predict(X[test])) / (y[test] + 1e-9))))
        for train, test in GroupKFold(n_splits=n_splits).split(X, y, groups)
    ]

    return {
        "beta_coefficients": {
            "beta_d_day":      float(model.coef_[0]),
            "beta_W_g":        float(model.coef_[1]),
            "beta_popularity": float(model.coef_[2]),
            "intercept":       float(model.intercept_),
        },
        "mu_base":       mu_base,
        "sigma":         sigma,
        "beta1_over_mu": beta1_over_mu,
        "logo_mape":     float(np.mean(mapes)),
    }


def run_regression_agent(state: dict) -> dict:
    print("[regression] Fitting WTP model...")
    seat_weights     = state["seat_weights"]
    popularity_score = state["concert_info"]["popularity_score"]
    errors           = list(state.get("errors", []))

    try:
        df = load_ticketbay_data()
    except Exception as e:
        errors.append(f"data_loader error: {e}")
        df = pd.DataFrame()

    try:
        X, y, groups = build_feature_matrix(df, seat_weights)
    except Exception as e:
        errors.append(f"feature matrix error: {e}")
        X, y, groups = np.zeros((0, 3)), np.zeros(0), np.zeros(0, int)

    if len(y) < 2:
        errors.append("regression: insufficient data — using mock fallback")
        # official_price 미사용 — 인기도별 티켓베이 시세 평균으로 fallback
        pop   = state["concert_info"].get("popularity_score", 4)
        mu_fb = {7: 165000, 6: 154000, 5: 132000, 4: 110000, 3: 99000, 2: 88000, 1: 77000}.get(pop, 110000)
        result = {
            "beta_coefficients": {"beta_d_day": -500.0, "beta_W_g": 20000.0, "beta_popularity": 5000.0, "intercept": float(mu_fb)},
            "mu_base": float(mu_fb), "sigma": float(mu_fb * 0.3),
            "beta1_over_mu": -0.003, "logo_mape": 0.0,
        }
    else:
        result = fit_wtp_model(X, y, groups)

    pop_factor             = _POPULARITY_FACTOR.get(popularity_score, 1.0)
    result["mu_final"]        = result["mu_base"] * pop_factor
    result["popularity_factor"] = pop_factor

    print(f"[regression] LOGO MAPE: {result['logo_mape']:.3f} | mu_final: {result['mu_final']:,.0f} KRW")
    save_json("regression.json", result)
    return {"wtp_model": result, "errors": errors}