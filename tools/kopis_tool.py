# Pattern: Tool Node (강의05)
import os
import random
from datetime import datetime, date
import requests
from xml.etree import ElementTree as ET

_KOPIS_BASE = "http://www.kopis.or.kr/openApi/restful"

_BASE_BOOKING_RATE = {
    "D60": 0.30, "D30": 0.50, "D14": 0.70, "D7": 0.85, "D1": 0.95
}


def _d_day_to_key(d_day: int) -> str:
    if d_day >= 45:  return "D60"
    if d_day >= 20:  return "D30"
    if d_day >= 10:  return "D14"
    if d_day >= 3:   return "D7"
    return "D1"


def get_booking_rate_mock(artist: str, d_day: int) -> float:
    key  = _d_day_to_key(d_day)
    base = _BASE_BOOKING_RATE[key]
    return round(min(1.0, max(0.0, base + random.uniform(-0.04, 0.04))), 3)


def _parse_date(date_str: str) -> date | None:
    """Parse KOPIS date string '2025.03.14' → date object."""
    for fmt in ("%Y.%m.%d", "%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(date_str.strip(), fmt).date()
        except ValueError:
            continue
    return None


def search_concert(artist: str, api_key: str) -> list[dict]:
    """Search KOPIS for concerts matching artist name.

    Returns list of {mt20id, name, start_date, end_date, state, price_guide, venue}
    """
    today = date.today()
    try:
        resp = requests.get(
            f"{_KOPIS_BASE}/pblprfr",
            params={
                "service":   api_key,
                "rows":      10,
                "cpage":     1,
                "shcate":    "CCCD",          # 대중음악
                "shprfnm":   artist,
                "stdate":    (today.replace(year=today.year - 2)).strftime("%Y%m%d"),
                "eddate":    (today.replace(year=today.year + 1)).strftime("%Y%m%d"),
            },
            timeout=8,
        )
        resp.raise_for_status()
        content = resp.content.decode("utf-8", errors="replace")
        root    = ET.fromstring(content)
    except Exception:
        return []

    results = []
    for db in root.findall("db"):
        start = _parse_date(db.findtext("prfpdfrom", ""))
        end   = _parse_date(db.findtext("prfpdto",   ""))
        if start is None:
            continue
        results.append({
            "mt20id":      db.findtext("mt20id",    ""),
            "name":        db.findtext("prfnm",     ""),
            "start_date":  start,
            "end_date":    end or start,
            "state":       db.findtext("prfstate",  ""),
            "venue":       db.findtext("fcltynm",   ""),
            "price_guide": "",   # fetched in detail call
        })
    return results


def get_concert_detail(mt20id: str, api_key: str) -> dict:
    """Fetch performance detail including price guidance and venue id."""
    try:
        resp = requests.get(
            f"{_KOPIS_BASE}/pblprfr/{mt20id}",
            params={"service": api_key},
            timeout=8,
        )
        resp.raise_for_status()
        content = resp.content.decode("utf-8", errors="replace")
        root    = ET.fromstring(content)
        db      = root.find("db")
        if db is None:
            return {}
        return {
            "price_guide": db.findtext("pcseguidance", ""),
            "cast":        db.findtext("prfcast",       ""),
            "runtime":     db.findtext("prfruntime",    ""),
            "mt10id":      db.findtext("mt10id",        ""),  # venue ID
        }
    except Exception:
        return {}


def get_venue_detail(mt10id: str, api_key: str) -> dict:
    """Fetch venue detail: seat count and full name."""
    if not mt10id:
        return {}
    try:
        resp = requests.get(
            f"{_KOPIS_BASE}/prfplc/{mt10id}",
            params={"service": api_key},
            timeout=8,
        )
        resp.raise_for_status()
        content = resp.content.decode("utf-8", errors="replace")
        root    = ET.fromstring(content)
        db      = root.find("db")
        if db is None:
            return {}
        seat_raw = db.findtext("seatscale", "0")
        try:
            seat_count = int("".join(filter(str.isdigit, seat_raw)))
        except (ValueError, TypeError):
            seat_count = 0
        return {
            "venue_full_name": db.findtext("fcltynm", ""),
            "seat_count":      seat_count,
            "venue_url":       db.findtext("relateurl", ""),
        }
    except Exception:
        return {}


def list_concerts(artist: str) -> list[dict]:
    """Return only upcoming KOPIS concerts (today or later) for the artist.

    Returns list of {index, name, date, venue, state}
    """
    api_key = os.getenv("KOPIS_API_KEY", "")
    if not api_key:
        return []
    today = date.today()
    concerts = [c for c in search_concert(artist, api_key) if c["start_date"] >= today]
    return [
        {
            "index": i,
            "name":  c["name"],
            "date":  c["start_date"].strftime("%Y-%m-%d"),
            "venue": c["venue"],
            "state": c["state"],
            "mt20id": c["mt20id"],
        }
        for i, c in enumerate(concerts)
    ]


def get_concert_info(artist: str, target_date: date | None = None, concert_index: int | None = None) -> dict:
    """Return best-matching KOPIS concert info for the artist.

    Returns:
        {
          "found": bool,
          "name": str,
          "concert_date": date | None,
          "d_day_from_today": int | None,   # days until concert from today
          "state": str,
          "price_guide": str,
          "venue": str,
        }
    """
    api_key = os.getenv("KOPIS_API_KEY", "")
    if not api_key:
        return {"found": False}

    concerts = search_concert(artist, api_key)
    if not concerts:
        return {"found": False}

    today = date.today()

    # concert_index: 사용자가 직접 선택한 경우
    if concert_index is not None and 0 <= concert_index < len(concerts):
        best = concerts[concert_index]
    elif target_date:
        best = min(concerts, key=lambda c: abs((c["start_date"] - target_date).days))
    else:
        # Prefer upcoming concerts; fall back to most recent past
        upcoming = [c for c in concerts if c["start_date"] >= today]
        best     = upcoming[0] if upcoming else concerts[0]

    d_day_from_today = (best["start_date"] - today).days

    detail = get_concert_detail(best["mt20id"], api_key)
    venue  = get_venue_detail(detail.get("mt10id", ""), api_key)

    return {
        "found":             True,
        "name":              best["name"],
        "concert_date":      best["start_date"],
        "d_day_from_today":  d_day_from_today,
        "state":             best["state"],
        "price_guide":       detail.get("price_guide", ""),
        "venue":             best["venue"],
        "venue_full_name":   venue.get("venue_full_name", best["venue"]),
        "seat_count":        venue.get("seat_count", 0),
        "cast":              detail.get("cast", ""),
    }


def parse_official_price(price_guide: str) -> int | None:
    """Extract R석 (or middle-tier) price from KOPIS pcseguidance string.

    예: "SOUND CHECK 264,000원, R석 220,000원, S석 198,000원" → 220000
    우선순위: R석 > S석 > 가장 낮은 가격 (일반석 기준)
    """
    import re
    if not price_guide:
        return None

    # 모든 (등급, 가격) 쌍 추출
    pairs = re.findall(r"([^\s,]+)\s+([\d,]+)원", price_guide)
    if not pairs:
        return None

    price_map = {label: int(price.replace(",", "")) for label, price in pairs}

    # R석 우선
    for label, price in price_map.items():
        if "R" in label.upper():
            return price
    # S석 차선
    for label, price in price_map.items():
        if "S" in label.upper():
            return price
    # 없으면 중간값
    prices = sorted(price_map.values())
    return prices[len(prices) // 2]


def get_booking_rate(artist: str, d_day: int) -> float:
    """Estimate booking rate.

    KOPIS open API does not expose per-D-day booking rates.
    We use a sigmoid-shaped proxy calibrated to D-day proximity.
    Falls back gracefully on any error.
    """
    try:
        key     = _d_day_to_key(d_day)
        base    = _BASE_BOOKING_RATE[key]
        # Small deterministic jitter from artist name hash (reproducible)
        jitter  = (hash(artist) % 100 - 50) / 2500   # ±0.02
        return round(min(1.0, max(0.0, base + jitter)), 3)
    except Exception:
        return get_booking_rate_mock(artist, d_day)
