from dotenv import load_dotenv
import pathlib
load_dotenv(dotenv_path=pathlib.Path(".env"), override=True)

from agents.seat_agent import _search_venue_image_url, assign_zone_variables

venue = "고척스카이돔"
print(f"검색 중: {venue}")
url = _search_venue_image_url(venue)
print(f"찾은 URL: {url}")

if url:
    print("\nClaude 분석 중...")
    result = assign_zone_variables("", venue_name=venue)
    import json
    print(json.dumps(result, indent=2, ensure_ascii=False))
