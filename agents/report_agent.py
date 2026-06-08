# Pattern: Sub-agent (강의06)
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import httpx
import anthropic
from scipy.stats import norm
from dotenv import load_dotenv
import pathlib

load_dotenv(dotenv_path=pathlib.Path(__file__).parent.parent / ".env", override=True)

from tools.simulation import (
    simulate_strategy1,
    simulate_strategy2,
    simulate_strategy3,
    calculate_kpi,
    find_optimal_static_price,
    simulate_sensitivity_scenarios,
)
from tools.file_manager import save_json

_INTERVALS   = ["D60", "D30", "D14", "D7", "D1"]
_REPORT_PATH = "results/report.png"


# ── F: Claude 전략 해설 ───────────────────────────────────────────────────────

def generate_insight(ci: dict, kpi: dict, lp_prices: dict, wtp: dict) -> str:
    """Ask Claude to explain the pricing strategy in plain language."""
    try:
        client = anthropic.Anthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            http_client=httpx.Client(verify=False),
        )
        price_summary = {
            iv: int(v["price"]) for iv, v in lp_prices.items()
        }
        prompt = (
            f"당신은 K-pop 콘서트 수익 분석 전문가입니다. "
            f"아래 동적 가격 책정 결과를 한국어로 3개의 짧은 단락으로 설명해주세요.\n\n"
            f"아티스트: {ci['artist']} | 인기도: {ci['popularity_score']}/7 | "
            f"WTP 기준가(mu_final): {wtp['mu_final']:,.0f}원 | 총 좌석: {ci['total_seats']:,}석\n"
            f"WTP 평균(μ): {wtp['mu_final']:,.0f}원 | 표준편차(σ): {wtp['sigma']:,.0f}원\n"
            f"D-day별 동적 가격: {price_summary}\n"
            f"고정가 대비 Revenue Gain: +{kpi['revenue_gain_pct']:.1f}%\n\n"
            f"다음 세 가지를 각각 한 단락씩 설명해주세요: "
            f"(1) D-day가 가까워질수록 가격이 이런 식으로 변하는 이유, "
            f"(2) WTP 분포가 이 아티스트 팬들의 특성에 대해 시사하는 점, "
            f"(3) 콘서트 주최사에게 드리는 실질적인 조언 한 가지."
        )
        msg = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text
    except Exception as e:
        return f"(Claude insight unavailable: {e})"


# ── G: 시각화 ─────────────────────────────────────────────────────────────────

def _setup_font():
    # 한국어 지원 폰트 우선순위로 탐색
    candidates = ["Malgun Gothic", "NanumGothic", "NanumBarunGothic", "Gulim", "Dotum"]
    available  = {f.name for f in fm.fontManager.ttflist}
    chosen     = next((f for f in candidates if f in available), None)
    if chosen:
        plt.rcParams["font.family"] = chosen
    else:
        # 폰트 없으면 한국어 → 영어로 대체되도록 경고 억제
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
    plt.rcParams["axes.unicode_minus"] = False


def _build_charts(
    s1: dict, s2: dict, s3: dict,
    kpi: dict, wtp: dict,
    artist: str, insight: str,
) -> str:
    _setup_font()

    fig = plt.figure(figsize=(18, 11))
    fig.suptitle(f"{artist} — Dynamic Pricing Analysis", fontsize=14, fontweight="bold", y=0.98)

    # Layout: 2x2 top + 1 bottom text row
    ax1 = fig.add_subplot(2, 3, 1)   # Revenue bar
    ax2 = fig.add_subplot(2, 3, 2)   # Price trajectory
    ax3 = fig.add_subplot(2, 3, 3)   # Demand curve (price elasticity)
    ax4 = fig.add_subplot(2, 3, 4)   # WTP distribution shift by D-day
    ax5 = fig.add_subplot(2, 3, 5)   # Revenue by D-day (strategy comparison)
    ax6 = fig.add_subplot(2, 3, 6)   # Claude insight text

    # ── Chart 1: Revenue bar ──────────────────────────────────────────────────
    rev_vals = [
        kpi["revenue_strategy1"] / 1e8,
        kpi["revenue_strategy2"] / 1e8,
        kpi["revenue_strategy3"] / 1e8,
    ]
    bars = ax1.bar(
        ["Strategy 1\n(Fixed)", "Strategy 2\n(Optimal\nStatic)", "Strategy 3\n(Dynamic LP)"],
        rev_vals,
        color=["#a8d8ea", "#f8b195", "#f67280"],
        edgecolor="white",
    )
    ax1.set_ylabel("Total Revenue (100M KRW)")
    ax1.set_title("Revenue Comparison")
    for i, v in enumerate(rev_vals):
        ax1.text(i, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    gain = kpi["revenue_gain_pct"]
    ax1.text(0.5, 0.97, f"Dynamic gain: +{gain:.1f}%", transform=ax1.transAxes,
             ha="center", va="top", fontsize=9, color="#c0392b", fontweight="bold")

    # ── Chart 2: Price trajectory ─────────────────────────────────────────────
    x = np.arange(len(_INTERVALS))
    for label, data, color in [
        ("Strategy 1 (Fixed)",   s1, "#a8d8ea"),
        ("Strategy 2 (Static)",  s2, "#f8b195"),
        ("Strategy 3 (Dynamic)", s3, "#f67280"),
    ]:
        ax2.plot(x, [data[iv]["price"] / 1000 for iv in _INTERVALS],
                 marker="o", label=label, color=color, linewidth=2)
    ax2.set_xticks(x)
    ax2.set_xticklabels(_INTERVALS)
    ax2.set_ylabel("Price (1,000 KRW)")
    ax2.set_title("Price Trajectory by D-day")
    ax2.legend(fontsize=7)
    ax2.grid(alpha=0.3)

    # ── Chart 3: Demand curve (price elasticity) — Strategy 3 at D14 ─────────
    mu    = wtp["mu_final"]
    sigma = wtp["sigma"]
    b1m   = wtp["beta1_over_mu"]
    d14_factor = max(0.5, min(1.0 + b1m * (14 - 14), 1.5))
    mu_d14 = mu * d14_factor

    price_range = np.linspace(mu_d14 * 0.4, mu_d14 * 2.2, 200)
    demand_curve = [float(s3.get("D14", s3.get("D7", {})).get("quantity", 1)) *
                    (1 - norm.cdf(p, mu_d14, sigma)) /
                    max(1 - norm.cdf(mu_d14 * 0.4, mu_d14, sigma), 1e-9)
                    for p in price_range]

    ax3.plot(price_range / 1000, demand_curve, color="#8e44ad", linewidth=2)
    ax3.axvline(s3.get("D14", {}).get("price", mu_d14) / 1000, color="#f67280",
                linestyle="--", linewidth=1.5, label="LP optimal price")
    ax3.set_xlabel("Price (1,000 KRW)")
    ax3.set_ylabel("Relative Demand")
    ax3.set_title("Demand Curve at D-14\n(WTP normal distribution)")
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.3)

    # ── Chart 4: WTP distribution shift across D-days ────────────────────────
    d_days = {"D60": 60, "D30": 30, "D14": 14, "D7": 7, "D1": 1}
    colors_wtp = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(_INTERVALS)))
    x_range = np.linspace(mu * 0.2, mu * 2.5, 300)
    for iv, col in zip(_INTERVALS, colors_wtp):
        df_val = max(0.5, min(1.0 + b1m * (14 - d_days[iv]), 1.5))
        mu_adj = mu * df_val
        pdf = norm.pdf(x_range, mu_adj, sigma)
        ax4.plot(x_range / 1000, pdf, label=iv, color=col, linewidth=1.8)
    ax4.set_xlabel("WTP (1,000 KRW)")
    ax4.set_ylabel("Density")
    ax4.set_title("WTP Distribution Shift by D-day")
    ax4.legend(fontsize=7)
    ax4.grid(alpha=0.3)

    # ── Chart 5: Revenue per D-day by strategy ────────────────────────────────
    x = np.arange(len(_INTERVALS))
    w = 0.25
    for i, (label, data, color) in enumerate([
        ("Fixed",   s1, "#a8d8ea"),
        ("Static",  s2, "#f8b195"),
        ("Dynamic", s3, "#f67280"),
    ]):
        ax5.bar(x + i * w, [data[iv]["revenue"] / 1e6 for iv in _INTERVALS],
                width=w, label=label, color=color, edgecolor="white")
    ax5.set_xticks(x + w)
    ax5.set_xticklabels(_INTERVALS)
    ax5.set_ylabel("Revenue (M KRW)")
    ax5.set_title("Revenue per D-day Interval")
    ax5.legend(fontsize=8)
    ax5.grid(alpha=0.3, axis="y")

    # ── Chart 6: Claude insight text ──────────────────────────────────────────
    ax6.axis("off")
    ax6.set_facecolor("#f9f9f9")
    wrapped = _wrap_text(insight, max_chars=72)
    ax6.text(0.04, 0.96, "AI Strategy Insight (Claude)", transform=ax6.transAxes,
             fontsize=9, fontweight="bold", va="top", color="#2c3e50")
    ax6.text(0.04, 0.88, wrapped, transform=ax6.transAxes,
             fontsize=7.5, va="top", color="#2c3e50",
             linespacing=1.5, wrap=True,
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#ecf0f1", alpha=0.6))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(_REPORT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return _REPORT_PATH


_SENSITIVITY_PATH = "results/sensitivity_report.png"


def _build_sensitivity_chart(
    scenarios: list,
    lp_base_price: float,
    mu_final: float,
    floor_price: float,
    artist: str,
) -> str:
    """가격 민감도 시나리오 분석 차트를 별도 PNG로 생성합니다.

    상단: 시나리오별 순수익 곡선 (price vs net revenue)
    하단: 시나리오별 최적가 / Revenue Gain 비교 표
    """
    _setup_font()
    import os
    os.makedirs("results", exist_ok=True)

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(13, 9),
        gridspec_kw={"height_ratios": [3, 1]},
    )
    fig.suptitle(
        f"{artist} — Price Sensitivity Scenario Analysis\n"
        f"(D-14 기준, ceiling 제거 — 이탈이 자연 천장 역할)",
        fontsize=13, fontweight="bold", y=0.99,
    )

    # ── 상단: 수익 곡선 ────────────────────────────────────────────────────────
    for sc in scenarios:
        xs  = np.array(sc["price_candidates"]) / 1000   # 천원 단위
        ys  = np.array(sc["net_revenues"])     / 1e8    # 억원 단위
        ax_top.plot(
            xs, ys,
            label=f"{sc['name']}  ({sc['label']})",
            color=sc["color"], linewidth=2.2,
        )
        # 최적가 별 표시
        opt_x = sc["optimal_price"] / 1000
        opt_y = sc["optimal_net_revenue"] / 1e8
        ax_top.scatter([opt_x], [opt_y], color=sc["color"], s=90, zorder=5)
        ax_top.annotate(
            f"  {opt_x:.0f}k",
            (opt_x, opt_y),
            fontsize=7.5, color=sc["color"], va="center",
        )

    # LP 현재 추천가 수직선 (S1 기준 = ceiling 있는 현재 모델)
    ax_top.axvline(
        lp_base_price / 1000,
        color="#7f8c8d", linestyle="--", linewidth=1.6,
        label=f"LP recommended (S1 ceiling)  {lp_base_price/1000:.0f}k",
    )
    # μ_final 기준선
    ax_top.axvline(
        mu_final / 1000,
        color="#bdc3c7", linestyle=":", linewidth=1.2,
        label=f"μ_final  {mu_final/1000:.0f}k",
    )

    ax_top.set_xlabel("Price (1,000 KRW)", fontsize=10)
    ax_top.set_ylabel("Net Revenue (100M KRW)", fontsize=10)
    ax_top.set_title("Revenue vs. Price under 5 Sensitivity Scenarios", fontsize=11)
    ax_top.legend(fontsize=8, loc="upper right")
    ax_top.grid(alpha=0.3)

    # ── 하단: 요약 표 ──────────────────────────────────────────────────────────
    ax_bot.axis("off")

    col_labels = ["Scenario", "Assumption", "Optimal Price", "Net Revenue (억)", "vs S1 (Gain)"]
    table_data = []
    for sc in scenarios:
        gain = sc["revenue_gain_vs_s1_pct"]
        gain_str = f"+{gain:.1f}%" if gain >= 0 else f"{gain:.1f}%  ◀ 손해"
        table_data.append([
            sc["name"],
            sc["label"],
            f"{sc['optimal_price']/1000:,.0f}k KRW",
            f"{sc['optimal_net_revenue']/1e8:.2f}",
            gain_str,
        ])

    tbl = ax_bot.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)

    # 헤더 스타일
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#2c3e50")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    # S4/S5 행 (브랜드 패널티 있는 시나리오) 배경 강조
    row_colors = ["#eafaf1", "#ebf5fb", "#fef9e7", "#fdebd0", "#fadbd8"]
    for i, color in enumerate(row_colors):
        for j in range(len(col_labels)):
            tbl[i + 1, j].set_facecolor(color)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(_SENSITIVITY_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return _SENSITIVITY_PATH


def _wrap_text(text: str, max_chars: int = 72) -> str:
    """Simple word-wrap for matplotlib text box."""
    words = text.split()
    lines, current = [], []
    length = 0
    for w in words:
        if length + len(w) + 1 > max_chars:
            lines.append(" ".join(current))
            current, length = [w], len(w)
        else:
            current.append(w)
            length += len(w) + 1
    if current:
        lines.append(" ".join(current))
    return "\n".join(lines)


# ── Node ──────────────────────────────────────────────────────────────────────

def run_report_agent(state: dict) -> dict:
    """LangGraph node function for report_node (Step 5 + 6)."""
    print("Step 5 📈 3전략 시뮬레이션 비교...")
    ci        = state["concert_info"]
    wtp       = state["wtp_model"]
    lp_prices = state["pricing_result"]
    cons      = state["constraints"]
    errors    = list(state.get("errors", []))

    # Strategy 1 baseline: WTP 중앙값(mu_final) 고정가 — official_price 불필요
    s1 = simulate_strategy1(int(wtp["mu_final"]), wtp, ci["total_seats"])
    static_price = find_optimal_static_price(cons["floor"], cons["ceiling"], wtp, ci["total_seats"])
    s2 = simulate_strategy2(static_price, wtp, ci["total_seats"])
    s3 = simulate_strategy3(lp_prices)

    kpi  = calculate_kpi(s1, s2, s3, int(wtp["mu_final"]), ci["total_seats"])
    gain = kpi["revenue_gain_pct"]
    print(f"Step 5 ✅ 전략3 Revenue Gain: +{gain:.1f}% vs 전략1 | MAPE: {kpi['mape']:.3f}")

    save_json("simulation.json", {"strategy1": s1, "strategy2": s2, "strategy3": s3})

    print("Step 6 🤖 Claude 전략 해설 생성 중...")
    insight = generate_insight(ci, kpi, lp_prices, wtp)

    report_path = _build_charts(s1, s2, s3, kpi, wtp, ci["artist"], insight)
    print(f"Step 6 📊 시각화 리포트 생성 완료 → {report_path}")

    # ── 가격 민감도 시나리오 분석 ────────────────────────────────────────────
    print("Step 6b 📉 가격 민감도 시나리오 분석 중...")
    scenarios = simulate_sensitivity_scenarios(
        wtp_model   = wtp,
        total_seats = ci["total_seats"],
        floor_price = float(cons["floor"]),
        d_day       = 14,
    )
    # D14 LP 추천가 (S1 기준 현재 모델)
    lp_d14_price = s3.get("D14", s3.get("D7", {})).get("price", wtp["mu_final"])
    sensitivity_path = _build_sensitivity_chart(
        scenarios     = scenarios,
        lp_base_price = float(lp_d14_price),
        mu_final      = wtp["mu_final"],
        floor_price   = float(cons["floor"]),
        artist        = ci["artist"],
    )
    save_json("sensitivity_scenarios.json", scenarios)
    print(f"Step 6b ✅ 시나리오 분석 완료 → {sensitivity_path}")

    return {"kpi": kpi, "report_path": report_path, "insight": insight, "errors": errors}
