"""KOPIS API response field inspection"""
from dotenv import load_dotenv
import pathlib, os, requests
from xml.etree import ElementTree as ET

load_dotenv(dotenv_path=pathlib.Path(".env"), override=True)
KEY = os.getenv("KOPIS_API_KEY", "")

# Concert list search (genre: popular music = CCCD)
resp = requests.get(
    "http://www.kopis.or.kr/openApi/restful/pblprfr",
    params={"service": KEY, "rows": 3, "cpage": 1, "shcate": "CCCD",
            "stdate": "20250101", "eddate": "20251231"},
    timeout=10,
)
content = resp.content.decode("utf-8", errors="replace")
root    = ET.fromstring(content)

print("=== Concert List ===")
print("raw:", content[:500])
for db in root.findall("db"):
    mt20id = db.findtext("mt20id", "")
    name   = db.findtext("prfnm", "")
    state  = db.findtext("prfstate", "")
    dates  = f"{db.findtext('prfpdfrom','')} ~ {db.findtext('prfpdto','')}"
    print(f"  {mt20id} | {name} | {state} | {dates}")

# Concert detail (first result)
first_id = root.find("db/mt20id")
if first_id is not None:
    resp2 = requests.get(
        f"http://www.kopis.or.kr/openApi/restful/pblprfr/{first_id.text}",
        params={"service": KEY},
        timeout=10,
    )
    root2 = ET.fromstring(resp2.text)
    print(f"\n=== Detail Fields ({first_id.text}) ===")
    for child in root2.find("db") or []:
        print(f"  {child.tag}: {child.text}")
