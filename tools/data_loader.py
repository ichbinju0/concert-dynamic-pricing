# Pattern: Tool Node (강의05)
# 실제 티켓베이 크롤링 데이터 + 인기도 데이터를 회귀분석에 맞는 형태로 전처리

import re
import numpy as np
import pandas as pd

import pathlib
_ROOT            = pathlib.Path(__file__).parent.parent  # pricing_agent/
_TICKETBAY_PATH  = str(_ROOT / "data" / "ticketbay_real.csv")
_POPULARITY_PATH = str(_ROOT / "data" / "artist_popularity_real.csv")
_MOCK_PATH       = str(_ROOT / "data" / "ticketbay_sample.csv")

# grade → seat_zone 매핑
# grade 없는 행은 "B" (중간 등급) 기본값 사용
_GRADE_MAP = {
    "VIP":   "A",
    "vip":   "A",
    "R":     "B",
    "r":     "B",
    "S":     "B",
    "s":     "B",
    "일반석": "C",
    "일반":   "C",
    "A":     "C",
    "a":     "C",
}

# d_day → kopis_booking_rate 근사값 (KOPIS API 없을 때)
def _approx_booking_rate(d_day: int) -> float:
    if d_day >= 45:
        base = 0.30
    elif d_day >= 20:
        base = 0.50
    elif d_day >= 10:
        base = 0.70
    elif d_day >= 3:
        base = 0.85
    else:
        base = 0.95
    rng = np.random.default_rng(seed=int(d_day))
    return round(float(np.clip(base + rng.uniform(-0.04, 0.04), 0.0, 1.0)), 3)


def _extract_artist(concert_name: str, known_artists: list[str]) -> str:
    """concert_name에서 알려진 아티스트 이름 추출.

    형식 예: "2026 양요섭 서울", "aespa 2024 KSPO Dome"
    known_artists에 있는 이름이 포함되면 그것을 반환.
    없으면 연도·지명을 제거한 나머지를 반환.
    """
    # 1) known_artists 중 concert_name에 포함된 것 찾기
    for artist in known_artists:
        if artist.lower() in concert_name.lower():
            return artist

    # 2) 연도(4자리 숫자)와 일반적인 지역명 제거 후 반환
    cleaned = re.sub(r"\b\d{4}\b", "", concert_name)
    cities  = ["서울", "부산", "대구", "인천", "광주", "대전", "수원", "성남",
               "Seoul", "Busan", "Incheon", "KSPO", "KINTEX", "COEX"]
    for city in cities:
        cleaned = cleaned.replace(city, "")
    return cleaned.strip()


def load_popularity_data(path: str = _POPULARITY_PATH) -> pd.DataFrame:
    """인기도 CSV 로드. 컬럼: artist_name, popularity_score"""
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="cp949")

    # 컬럼 이름 정규화
    col_map = {}
    for col in df.columns:
        if "그룹" in col or "가수" in col or "아티스트" in col or "name" in col.lower():
            col_map[col] = "artist_name"
        elif "인기도" in col or "popularity" in col.lower() or "score" in col.lower():
            col_map[col] = "popularity_score"
    df = df.rename(columns=col_map)

    needed = {"artist_name", "popularity_score"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"인기도 파일에 필요한 컬럼 없음: {missing}\n실제 컬럼: {list(df.columns)}")

    df["popularity_score"] = pd.to_numeric(df["popularity_score"], errors="coerce").fillna(4).astype(int)
    return df[["artist_name", "popularity_score"]].dropna()


def load_ticketbay_data(
    ticketbay_path: str = _TICKETBAY_PATH,
    popularity_path: str = _POPULARITY_PATH,
) -> pd.DataFrame:
    """티켓베이 크롤링 데이터를 회귀분석용 DataFrame으로 변환.

    반환 컬럼:
        artist_name, artist_group, popularity_score, official_price,
        d_day, seat_zone, listing_price, kopis_booking_rate
    """
    # ── 파일 로드 ──────────────────────────────────────────────────
    try:
        df = pd.read_csv(ticketbay_path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(ticketbay_path, encoding="cp949")
    except FileNotFoundError:
        print(f"[data_loader] {ticketbay_path} 없음 → mock 데이터 사용")
        ticketbay_path = _MOCK_PATH
        try:
            df = pd.read_csv(ticketbay_path, encoding="utf-8-sig")
        except UnicodeDecodeError:
            df = pd.read_csv(ticketbay_path, encoding="cp949")

    try:
        pop_df = load_popularity_data(popularity_path)
        known_artists = pop_df["artist_name"].tolist()
    except Exception:
        pop_df = pd.DataFrame(columns=["artist_name", "popularity_score"])
        known_artists = []

    # ── 필수 컬럼 확인 ─────────────────────────────────────────────
    required = {"concert_name", "d_day", "resale_price", "official_price"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"티켓베이 파일에 필요한 컬럼 없음: {missing}\n실제 컬럼: {list(df.columns)}")

    result = pd.DataFrame()

    # ── listing_price (WTP proxy) ──────────────────────────────────
    result["listing_price"]  = pd.to_numeric(df["resale_price"], errors="coerce")
    result["official_price"] = pd.to_numeric(df["official_price"], errors="coerce")

    # d_day: "D-1", "D-7" 같은 문자열 → 숫자 추출
    raw_dday = df["d_day"].astype(str).str.extract(r"(\d+)")[0]
    result["d_day"] = pd.to_numeric(raw_dday, errors="coerce").abs()

    # ── seat_zone ─────────────────────────────────────────────────
    # grade 있으면 grade 우선, 없으면 area, 둘 다 없으면 "B"
    if "grade" in df.columns:
        grade_series = df["grade"].astype(str).str.strip()
        result["seat_zone"] = grade_series.map(_GRADE_MAP)
    else:
        result["seat_zone"] = pd.NA

    if "area" in df.columns and result["seat_zone"].isna().any():
        area_series = df["area"].astype(str).str.strip()
        area_mapped = area_series.map(_GRADE_MAP)
        result["seat_zone"] = result["seat_zone"].fillna(area_mapped)

    result["seat_zone"] = result["seat_zone"].fillna("B")  # 최종 fallback

    # ── artist_name 추출 ───────────────────────────────────────────
    result["artist_name"] = df["concert_name"].astype(str).apply(
        lambda x: _extract_artist(x, known_artists)
    )

    # ── popularity_score 합치기 ────────────────────────────────────
    if not pop_df.empty:
        result = result.merge(pop_df, on="artist_name", how="left")
    if "popularity_score" not in result.columns:
        result["popularity_score"] = 4  # 중간값 fallback
    result["popularity_score"] = result["popularity_score"].fillna(4).astype(int)

    # ── artist_group (LOGO CV용 숫자 ID) ──────────────────────────
    unique_artists = result["artist_name"].unique()
    artist_to_group = {name: i + 1 for i, name in enumerate(sorted(unique_artists))}
    result["artist_group"] = result["artist_name"].map(artist_to_group)

    # ── kopis_booking_rate (d_day 기반 근사) ──────────────────────
    result["kopis_booking_rate"] = result["d_day"].apply(
        lambda x: _approx_booking_rate(int(x)) if pd.notna(x) else 0.5
    )

    # ── 결측값 제거 ───────────────────────────────────────────────
    result = result.dropna(subset=["listing_price", "official_price", "d_day"])
    result = result[result["listing_price"] > 0]
    result = result[result["d_day"] >= 0]

    print(f"[data_loader] 로드 완료: {len(result)}행 / {result['artist_name'].nunique()}개 아티스트")
    print(f"[data_loader] seat_zone 분포:\n{result['seat_zone'].value_counts().to_string()}")

    return result.reset_index(drop=True)
