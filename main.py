import pathlib
from dotenv import load_dotenv

load_dotenv(dotenv_path=pathlib.Path(__file__).parent / ".env", override=True)

from graph.state import PricingState
from graph.pipeline import build_pipeline
from tools.constraint_calibrator import followers_to_popularity
from tools.kopis_tool import list_concerts


def run_agent(concert_input: dict) -> dict:
    ci = dict(concert_input)
    if "followers_in_10k" in ci and "popularity_score" not in ci:
        score = followers_to_popularity(ci.pop("followers_in_10k"))
        ci["popularity_score"] = score
        print(f"[info] Followers {concert_input['followers_in_10k']}M → Popularity score {score}")

    initial_state: PricingState = {
        "concert_info":   ci,
        "constraints":    {},
        "seat_weights":   {},
        "wtp_model":      {},
        "pricing_result": {},
        "kpi":            {},
        "insight":        "",
        "report_path":    "",
        "current_step":   "input",
        "errors":         [],
    }
    app = build_pipeline()
    return app.invoke(initial_state)


if __name__ == "__main__":
    ARTIST           = input("Artist: ").strip()
    FOLLOWERS_IN_10K = float(input("Instagram followers (unit: 10k, e.g. 281.8): ").strip())

    extra = {}

    print(f"\n[search] Searching KOPIS for '{ARTIST}'...")
    concerts = list_concerts(ARTIST)
    concert_index = None

    if concerts:
        print("\n[concerts] Upcoming concerts:")
        for c in concerts:
            print(f"  [{c['index']}] {c['date']}  {c['name']}  @ {c['venue']}")
        try:
            choice = input("\nSelect concert number (Enter = auto): ").strip()
            concert_index = int(choice) if choice else None
        except (ValueError, EOFError):
            concert_index = None
    else:
        print(f"\n[warning] No upcoming concerts found for '{ARTIST}' on KOPIS.")
        print("Please enter concert details manually.\n")

        venue            = input("Venue name (e.g. Olympic Park): ").strip()
        sale_start_d_day = int(input("Days until concert (e.g. 60): ").strip())
        total_seats      = input("Total seats (Enter to skip): ").strip()
        image_path       = input("Seating chart image path (Enter to skip): ").strip()

        extra["venue"]            = venue
        extra["sale_start_d_day"] = sale_start_d_day
        extra["skip_kopis"]       = True
        if total_seats:
            extra["total_seats"]      = int(total_seats)
        if image_path:
            extra["venue_image_path"] = image_path

    IMAGE_PATH = input("Venue image path (e.g. data/day6.png, Enter = skip): ").strip()
    if not IMAGE_PATH:
        IMAGE_PATH = None

    result = run_agent({
        "artist":           ARTIST,
        "followers_in_10k": FOLLOWERS_IN_10K,
        "concert_index":    concert_index,
        **({"venue_image_path": IMAGE_PATH} if IMAGE_PATH else {}),
        **extra,
    })

    kpi  = result.get("kpi", {})
    lp   = result.get("pricing_result", {})
    cons = result.get("constraints", {})

    print(f"\n{'='*50}")
    print(f"Revenue Gain : +{kpi.get('revenue_gain_pct', 0):.1f}%")
    print(f"MAPE         : {kpi.get('mape', 0):.3f}")
    print(f"Floor        : {cons.get('floor', 0):,} KRW")
    print(f"Ceiling      : {cons.get('ceiling', 0):,} KRW")
    print(f"\nRecommended prices by D-day:")
    for interval, v in lp.items():
        print(f"\n  [{interval}] Base {int(v['price']):,} KRW  (demand {v['quantity']:.0f} seats)")
        for zone, price in v.get("zone_prices", {}).items():
            print(f"    {zone:<20}: {price:>10,} KRW")

    print(f"\nReport : {result.get('report_path', 'N/A')}")
    if result.get("errors"):
        print(f"Errors : {result['errors']}")
