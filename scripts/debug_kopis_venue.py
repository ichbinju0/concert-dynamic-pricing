"""KOPIS venue API - seat chart image lookup"""
from dotenv import load_dotenv
import pathlib, os, requests
from xml.etree import ElementTree as ET

load_dotenv(dotenv_path=pathlib.Path(".env"), override=True)
KEY = os.getenv("KOPIS_API_KEY", "")

# Venue detail lookup by mt10id
mt10id = "FC001169"
resp = requests.get(
    f"http://www.kopis.or.kr/openApi/restful/prfplc/{mt10id}",
    params={"service": KEY},
    timeout=10,
)
root = ET.fromstring(resp.content.decode("utf-8", errors="replace"))
db   = root.find("db")

print("=== Venue Detail Fields ===")
if db is not None:
    for child in db:
        print(f"  {child.tag}: {child.text}")

# Search venues by name
resp2 = requests.get(
    "http://www.kopis.or.kr/openApi/restful/prfplc",
    params={"service": KEY, "rows": 5, "cpage": 1, "shprfnm": "KSPO"},
    timeout=10,
)
root2 = ET.fromstring(resp2.content.decode("utf-8", errors="replace"))

print("\n=== KSPO Venue Search ===")
for db2 in root2.findall("db"):
    mt10id2 = db2.findtext("mt10id", "")
    name2   = db2.findtext("fcltynm", "")
    print(f"  {mt10id2} | {name2}")
    for child in db2:
        if child.text and "http" in str(child.text):
            print(f"    [image] {child.tag}: {child.text}")