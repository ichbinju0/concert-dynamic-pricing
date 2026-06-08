"""Run once: python data/generate_mock.py"""
import csv
import random
import os

random.seed(42)

ARTISTS = [
    {"name": "IU",     "group": 1, "popularity": 7, "official_price": 165000},
    {"name": "aespa",  "group": 2, "popularity": 6, "official_price": 154000},
    {"name": "TWS",    "group": 3, "popularity": 4, "official_price": 110000},
    {"name": "LABOUM", "group": 4, "popularity": 2, "official_price": 88000},
]
D_DAYS = [60, 30, 14, 7, 1]
ZONES  = ["A", "B", "C"]

POPULARITY_MULT = {7: 1.90, 6: 1.55, 4: 1.10, 2: 0.92}
ZONE_MULT       = {"A": 1.30, "B": 1.00, "C": 0.75}
D_DAY_MULT      = {60: 0.85, 30: 0.92, 14: 1.00, 7: 1.10, 1: 1.25}
BOOKING_RATE    = {60: 0.30, 30: 0.50, 14: 0.70, 7: 0.85, 1: 0.95}

script_dir = os.path.dirname(os.path.abspath(__file__))

rows = []
for artist in ARTISTS:
    for d_day in D_DAYS:
        for zone in ZONES:
            for _ in range(3):
                noise   = random.uniform(0.93, 1.07)
                listing = int(
                    artist["official_price"]
                    * POPULARITY_MULT[artist["popularity"]]
                    * ZONE_MULT[zone]
                    * D_DAY_MULT[d_day]
                    * noise
                )
                rows.append({
                    "artist_name":        artist["name"],
                    "artist_group":       artist["group"],
                    "popularity_score":   artist["popularity"],
                    "official_price":     artist["official_price"],
                    "d_day":              d_day,
                    "seat_zone":          zone,
                    "listing_price":      listing,
                    "kopis_booking_rate": round(
                        BOOKING_RATE[d_day] + random.uniform(-0.03, 0.03), 3
                    ),
                })

fieldnames = [
    "artist_name", "artist_group", "popularity_score", "official_price",
    "d_day", "seat_zone", "listing_price", "kopis_booking_rate",
]
ticketbay_path = os.path.join(script_dir, "ticketbay_sample.csv")
with open(ticketbay_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
print(f"Written {len(rows)} rows to ticketbay_sample.csv")

popularity_path = os.path.join(script_dir, "artist_popularity.csv")
with open(popularity_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["artist_name", "popularity_score"])
    writer.writeheader()
    for a in ARTISTS:
        writer.writerow({"artist_name": a["name"], "popularity_score": a["popularity"]})
print("Written artist_popularity.csv")
