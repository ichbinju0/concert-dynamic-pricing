# Pattern: Tool Node (강의05)
# floor/ceiling ratios calibrated from observed (ticketbay listing_price / official_price)
# distribution by artist popularity tier — based on 4-group dataset collected May 2026.

# Follower brackets (unit: 만명) → popularity score 1-7
# Source: artist_popularity_real.csv 평가표
_FOLLOWER_BRACKETS = [
    (0,    10,   1),
    (10,   25,   2),
    (25,   40,   3),
    (40,   100,  4),
    (100,  250,  5),
    (250,  1000, 6),
    (1000, float("inf"), 7),
]


def followers_to_popularity(followers_in_10k: float) -> int:
    """Convert Instagram follower count (단위: 만명) to popularity score 1-7.

    Args:
        followers_in_10k: 팔로워 수 (만 단위). 예: 150 → 150만명

    Returns:
        popularity_score: 1-7 정수
    """
    for lo, hi, score in _FOLLOWER_BRACKETS:
        if lo <= followers_in_10k < hi:
            return score
    return 7


# 인기도별 WTP sigma 배수 — 높은 인기도일수록 ceiling을 더 넓게 허용
_SIGMA_BOUNDS = {
    7: {"floor": 1.5, "ceiling": 2.5},
    6: {"floor": 1.5, "ceiling": 2.2},
    5: {"floor": 1.3, "ceiling": 2.0},
    4: {"floor": 1.2, "ceiling": 1.8},
    3: {"floor": 1.0, "ceiling": 1.5},
    2: {"floor": 0.8, "ceiling": 1.3},
    1: {"floor": 0.8, "ceiling": 1.2},
}


def get_price_bounds_from_wtp(mu_final: float, sigma: float, popularity_score: int) -> dict:
    """Derive floor/ceiling directly from the learned WTP distribution.

    floor   = mu_final - floor_sigma  × sigma
    ceiling = mu_final + ceiling_sigma × sigma

    Args:
        mu_final: adjusted WTP mean (KRW) from regression
        sigma:    WTP standard deviation (KRW) from regression
        popularity_score: 1-7

    Returns:
        {"floor": int, "ceiling": int}
    """
    bounds    = _SIGMA_BOUNDS.get(popularity_score, _SIGMA_BOUNDS[4])
    floor_p   = max(10000, int(mu_final - bounds["floor"]   * sigma))
    ceiling_p = int(mu_final + bounds["ceiling"] * sigma)
    print(
        f"Step 3 ✅ WTP 기반 가격 범위 설정 → "
        f"Floor: {floor_p:,}원 / Ceiling: {ceiling_p:,}원 "
        f"(μ={mu_final:,.0f} σ={sigma:,.0f})"
    )
    return {"floor": floor_p, "ceiling": ceiling_p}
