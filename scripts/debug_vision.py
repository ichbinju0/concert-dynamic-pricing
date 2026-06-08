from dotenv import load_dotenv
import pathlib, os, base64, httpx, anthropic, json, re

load_dotenv(dotenv_path=pathlib.Path(".env"), override=True)

from agents.seat_agent import _ZONE_PROMPT_SUFFIX

with open("data/venue.jpg", "rb") as f:
    header = f.read(8)
    f.seek(0)
    b64 = base64.standard_b64encode(f.read()).decode("ascii")

mime   = "image/png" if header[:8] == b"\x89PNG\r\n\x1a\n" else "image/jpeg"
client = anthropic.Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    http_client=httpx.Client(verify=False),
)

msg = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=4096,
    messages=[{
        "role": "user",
        "content": [
            {"type": "image", "source": {"type": "base64", "media_type": mime, "data": b64}},
            {"type": "text", "text": f"This is a K-pop concert venue seating chart. The stage or main performance area is marked in the image.{_ZONE_PROMPT_SUFFIX}"},
        ],
    }],
)

raw = re.sub(r"```(?:json)?\s*", "", msg.content[0].text).strip()
print(f"Zones found: {raw.count('Z1')}")
print(raw[:2000])