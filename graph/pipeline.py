# Pattern: LangGraph StateGraph (강의07)
from langgraph.graph import StateGraph, END

from graph.state import PricingState
from tools.constraint_calibrator import get_price_bounds_from_wtp
from tools.kopis_tool import get_concert_info, list_concerts
from agents.seat_agent import run_seat_agent
from models.regression import run_regression_agent
from models.lp_solver import run_lp_agent
from agents.report_agent import run_report_agent


def input_node(state: PricingState) -> dict:
    """Step 1: Echo concert info + KOPIS lookup (auto-fills venue name & seat count)."""
    ci     = dict(state["concert_info"])
    errors = []

    print(f"\n🎵 Dynamic Pricing Agent 시작")
    print(f"Step 1 ✅ 콘서트 정보 입력: {ci['artist']} @ {ci.get('venue', '미정')}")

    concert_index = ci.pop("concert_index", None)
    skip_kopis    = ci.pop("skip_kopis", False)   # 수동 입력 경로면 True

    kopis = {} if skip_kopis else get_concert_info(ci["artist"], concert_index=concert_index)
    if not skip_kopis and kopis.get("found"):
        print(f"Step 1 🔍 KOPIS 검색 결과: {kopis['name']}")
        print(f"         날짜: {kopis['concert_date']}  |  D-{kopis['d_day_from_today']}  |  상태: {kopis['state']}")
        if kopis.get("price_guide"):
            print(f"         가격 안내: {kopis['price_guide']}")

        # ── 공연장명 자동 주입 ──────────────────────────────────────────
        if kopis.get("venue_full_name"):
            ci["kopis_venue_name"] = kopis["venue_full_name"]
            if not ci.get("venue"):
                ci["venue"] = kopis["venue_full_name"]
            print(f"         공연장: {kopis['venue_full_name']}")

        # ── 좌석 수 자동 주입 ──────────────────────────────────────────
        if kopis.get("seat_count", 0) > 0:
            if not ci.get("total_seats"):
                ci["total_seats"] = kopis["seat_count"]
                print(f"         좌석 수: {kopis['seat_count']:,}석 (KOPIS 자동)")
            else:
                print(f"         좌석 수: {ci['total_seats']:,}석 (입력값) / KOPIS: {kopis['seat_count']:,}석")

        # ── 가격 안내 참고용 출력 (floor/ceiling은 WTP에서 자동 도출) ────
        if kopis.get("price_guide"):
            print(f"         KOPIS 가격 안내 (참고): {kopis['price_guide']}")

        # ── D-day 자동 계산 ────────────────────────────────────────────
        if kopis.get("d_day_from_today") is not None:
            kopis_dday = max(1, kopis["d_day_from_today"])
            if not ci.get("sale_start_d_day"):
                ci["sale_start_d_day"] = kopis_dday
                print(f"         D-day: D-{kopis_dday} (KOPIS 자동)")
            else:
                print(f"         D-day: D-{ci['sale_start_d_day']} (입력값) / 실제 공연: D-{kopis_dday}")
    elif skip_kopis:
        print(f"Step 1 ✅ 수동 입력값 사용 (KOPIS 조회 건너뜀)")
    else:
        print(f"Step 1 ℹ️  KOPIS에서 '{ci['artist']}' 공연을 찾지 못했습니다 (입력값 사용)")

    # ── 필수값 최종 fallback ───────────────────────────────────────────────
    if not ci.get("total_seats"):
        ci["total_seats"] = 10000
        print(f"         좌석 수: 10,000석 (기본값 — 직접 입력 권장)")
    if not ci.get("sale_start_d_day"):
        ci["sale_start_d_day"] = 60
        print(f"         D-day: 60 (기본값)")

    return {"concert_info": ci, "current_step": "seat", "errors": errors}


def seat_node(state: PricingState) -> dict:
    """Step 2: Claude API + OLS W_g (constraints set after regression)."""
    seat_result = run_seat_agent(state)
    return {**seat_result, "current_step": "regression"}


def regression_node(state: PricingState) -> dict:
    """Step 3: WTP regression → derive floor/ceiling from learned distribution."""
    result = run_regression_agent(state)
    wtp    = result["wtp_model"]
    pop    = state["concert_info"]["popularity_score"]
    constraints = get_price_bounds_from_wtp(wtp["mu_final"], wtp["sigma"], pop)
    return {**result, "constraints": constraints, "current_step": "lp"}


def lp_node(state: PricingState) -> dict:
    result = run_lp_agent(state)
    return {**result, "current_step": "report"}


def report_node(state: PricingState) -> dict:
    result = run_report_agent(state)
    return {**result, "current_step": "done"}


def build_pipeline():
    """Compile and return the LangGraph app."""
    graph = StateGraph(PricingState)

    graph.add_node("input_node",      input_node)
    graph.add_node("seat_node",       seat_node)
    graph.add_node("regression_node", regression_node)
    graph.add_node("lp_node",         lp_node)
    graph.add_node("report_node",     report_node)

    graph.set_entry_point("input_node")
    graph.add_edge("input_node",      "seat_node")
    graph.add_edge("seat_node",       "regression_node")
    graph.add_edge("regression_node", "lp_node")
    graph.add_edge("lp_node",         "report_node")
    graph.add_edge("report_node",     END)

    return graph.compile()
