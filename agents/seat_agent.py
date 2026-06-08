# Pattern: Sub-agent + Claude API Tool Use (강의06)
import base64
import json
import mimetypes
import os
import re

import anthropic
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from tools.file_manager import save_json

_FALLBACK_BETAS = {"beta_Z1": 0.4, "beta_Z2": 0.4, "beta_Z3": 0.2}
import pathlib as _pl
_DATA_PATH = str(_pl.Path(__file__).parent.parent / "data" / "ticketbay_real.csv")

_ZONE_PROMPT_SUFFIX = (
    "\nCarefully read ALL section labels visible in this seating chart image."
    " Include every single section — do not skip any, even if there are 50+."
    " Use the EXACT label printed in the image (e.g. E1, E2, W3, S11, N7, VIP2, FLOOR, 101)."
    " For each section assign three hedonic scores based on its position:"
    " Z1 = stage/pitch proximity (0.0=farthest row, 1.0=right next to stage/pitch),"
    " Z2 = frontality (0.0=behind stage or pure side view, 1.0=directly facing stage/pitch center),"
    " Z3 = runway/pit adjacent (1 if touching the runway/pit/floor area, else 0)."
    " Sections at the same distance from stage but different angles should differ in Z2."
    " Return ONLY a JSON object, no markdown fences, no explanation:"
    ' {"zones":{"E1":{"Z1":0.0,"Z2":0.0,"Z3":0},"E2":{...},...}}'
)

# For venue-name mode (no image): aggregate into tiers is fine
_ZONE_PROMPT_SUFFIX_TIERS = (
    "\nGroup the seating areas into major tiers (e.g. VIP, FLOOR, LOWER, UPPER, SIDE)."
    " Return JSON only, no other text, no explanation."
    ' Format: {"zones":{"TierName":{"Z1":0.0,"Z2":0.0,"Z3":0},...}}'
    " Z1: stage proximity (0=far, 1=closest),"
    " Z2: frontality (0=side view, 1=directly facing stage),"
    " Z3: runway adjacent (1=yes, 0=no)."
)


def _build_messages_text(venue_description: str) -> list:
    """Text-only message for Claude."""
    venue_safe = json.dumps(venue_description, ensure_ascii=True)[1:-1]
    return [{
        "role": "user",
        "content": f"Read the following concert venue description and assign hedonic variables.\n\nVenue description: {venue_safe}\n{_ZONE_PROMPT_SUFFIX}",
    }]


def _search_venue_image_url(venue_name: str) -> str | None:
    """Search DuckDuckGo Images for venue seating chart and return first image URL."""
    try:
        from ddgs import DDGS
        query = f"{venue_name} 좌석배치도"
        with DDGS() as ddgs:
            results = list(ddgs.images(query, max_results=5))
        for r in results:
            url = r.get("image", "")
            if url and url.startswith("https"):
                return url
    except Exception:
        pass
    return None


def _build_messages_venue_image_url(venue_name: str, image_url: str) -> list:
    """Pass venue seating chart image via URL to Claude Vision."""
    return [{
        "role": "user",
        "content": [
            {
                "type": "image",
                "source": {"type": "url", "url": image_url},
            },
            {
                "type": "text",
                "text": (
                    f"This is the seating chart for '{venue_name}'."
                    f"{_ZONE_PROMPT_SUFFIX}"
                ),
            },
        ],
    }]


def _build_messages_venue_name(venue_name: str) -> list:
    """Ask Claude to infer zone layout from well-known venue name alone (no image)."""
    venue_safe = json.dumps(venue_name, ensure_ascii=True)[1:-1]
    return [{
        "role": "user",
        "content": (
            f"You are a K-pop concert seating expert. "
            f"Based on your knowledge of the concert venue named '{venue_safe}', "
            f"infer the typical seating zone layout and assign hedonic variables. "
            f"If you know this venue, describe its actual zones. "
            f"If unknown, assume a standard arena layout with floor/lower/upper tiers."
            f"\n{_ZONE_PROMPT_SUFFIX_TIERS}"
        ),
    }]


def _build_messages_image(image_path: str, hint: str = "") -> list:
    """Vision message: encode local image file as base64 for Claude.

    hint: optional short description (e.g. zone names, special layout notes)
    """
    # Detect actual format from file header, not filename extension
    with open(image_path, "rb") as _f:
        header = _f.read(8)
    if header[:8] == b'\x89PNG\r\n\x1a\n':
        mime = "image/png"
    elif header[:3] == b'\xff\xd8\xff':
        mime = "image/jpeg"
    elif header[:6] in (b'GIF87a', b'GIF89a'):
        mime = "image/gif"
    elif header[:4] == b'RIFF':
        mime = "image/webp"
    else:
        mime, _ = mimetypes.guess_type(image_path)
        if mime not in ("image/jpeg", "image/png", "image/gif", "image/webp"):
            mime = "image/png"
    with open(image_path, "rb") as f:
        b64 = base64.standard_b64encode(f.read()).decode("ascii")
    hint_text = f" Context: {hint}." if hint else ""
    return [{
        "role": "user",
        "content": [
            {
                "type": "image",
                "source": {"type": "base64", "media_type": mime, "data": b64},
            },
            {
                "type": "text",
                "text": (
                    f"This is a K-pop concert venue seating chart.{hint_text}"
                    f" The stage or main performance area is marked in the image."
                    f"{_ZONE_PROMPT_SUFFIX}"
                ),
            },
        ],
    }]


def assign_zone_variables(
    venue_description: str,
    image_path: str | None = None,
    venue_name: str = "",
    max_retries: int = 3,
) -> dict:
    """Ask Claude to assign Z1/Z2/Z3 per zone.

    If image_path is provided, Claude Vision analyzes the seating chart image.
    Otherwise falls back to text description.
    """
    import httpx
    # SSL verification disabled: school/corp network uses SSL inspection proxy
    http_client = httpx.Client(verify=False)
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"), http_client=http_client)

    effective_venue = venue_name or venue_description

    if image_path and os.path.exists(image_path):
        # 1순위: 직접 제공한 이미지 파일
        messages = _build_messages_image(image_path, hint=venue_description)
        print(f"  [vision] 이미지 분석 중: {os.path.basename(image_path)}")
    elif venue_description:
        # 2순위: 텍스트 설명
        messages = _build_messages_text(venue_description)
        print(f"  [text] 텍스트 설명으로 분석 중")
    elif venue_name:
        # 3순위: 공연장명으로 Claude가 레이아웃 추론
        messages = _build_messages_venue_name(venue_name)
        print(f"  [venue-name] 공연장명으로 분석: {venue_name}")
    else:
        messages = _build_messages_venue_name("standard K-pop indoor arena")
        print(f"  [venue-name] 공연장 정보 없음 → 기본 아레나 레이아웃 사용")

    for attempt in range(max_retries):
        try:
            msg = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=4096,   # 개별 구역 수십 개 대응
                messages=messages,
            )
            raw = msg.content[0].text
            # Strip markdown code fences (```json ... ``` or ``` ... ```)
            raw = re.sub(r"```(?:json)?\s*", "", raw).strip()
            json_match = re.search(r"\{.*\}", raw, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except (json.JSONDecodeError, AttributeError, IndexError) as e:
            if attempt == max_retries - 1:
                raise RuntimeError(
                    f"Claude API JSON parsing failed after {max_retries} tries: {e}"
                )
    return {}


def compute_hedonic_weights(zone_vars: dict, zone_prices: dict | None) -> dict:
    """Step B: OLS on zone-level prices -> beta_hedonic -> W_g per zone.

    Falls back to preset betas when zone_prices is unavailable.
    """
    if zone_prices is None or len(zone_prices) < 2:
        betas = _FALLBACK_BETAS
    else:
        zones = list(zone_vars.keys())
        X = np.array(
            [[zone_vars[z]["Z1"], zone_vars[z]["Z2"], zone_vars[z]["Z3"]] for z in zones]
        )
        y = np.array([zone_prices[z] for z in zones if z in zone_prices])
        if len(y) < 2:
            betas = _FALLBACK_BETAS
        else:
            model = LinearRegression(fit_intercept=False)
            model.fit(X, y)
            betas = {
                "beta_Z1": float(model.coef_[0]),
                "beta_Z2": float(model.coef_[1]),
                "beta_Z3": float(model.coef_[2]),
            }

    result_zones = {}
    for zone, zvars in zone_vars.items():
        w_g = (
            betas["beta_Z1"] * zvars["Z1"]
            + betas["beta_Z2"] * zvars["Z2"]
            + betas["beta_Z3"] * zvars["Z3"]
        )
        result_zones[zone] = {**zvars, "W_g": round(float(w_g), 4)}

    return {"hedonic_betas": betas, "zones": result_zones}


def _load_zone_prices() -> dict | None:
    """Load mean listing price per seat_zone from ticketbay CSV."""
    try:
        df = pd.read_csv(_DATA_PATH)
        if "seat_zone" not in df.columns:
            return None
        return df.groupby("seat_zone")["listing_price"].mean().to_dict()
    except FileNotFoundError:
        return None


def run_seat_agent(state: dict) -> dict:
    """LangGraph node function for seat_node (Step 2)."""
    ci            = state["concert_info"]
    venue_desc    = ci.get("venue_description", "")
    image_path    = ci.get("venue_image_path")
    venue_name    = ci.get("kopis_venue_name") or ci.get("venue", "")
    errors        = list(state.get("errors", []))

    print(f"Step 2 🤖 Claude API 좌석 변수 할당 중...")

    try:
        zone_vars_raw = assign_zone_variables(
            venue_desc, image_path=image_path, venue_name=venue_name
        )
        zone_vars = zone_vars_raw.get("zones", {})
    except Exception as e:
        errors.append(f"seat_agent Claude API error: {e}")
        zone_vars = {
            "A": {"Z1": 0.9, "Z2": 0.9, "Z3": 1},
            "B": {"Z1": 0.5, "Z2": 0.6, "Z3": 0},
            "C": {"Z1": 0.2, "Z2": 0.3, "Z3": 0},
        }

    zone_prices = _load_zone_prices()
    weights = compute_hedonic_weights(zone_vars, zone_prices)

    zone_count = len(weights["zones"])
    print(f"Step 2 ✅ {zone_count}개 구역 헤도닉 변수 할당 완료 + W_g 계산 완료")

    save_json("seat_variables.json", weights)
    return {"seat_weights": weights, "errors": errors}
