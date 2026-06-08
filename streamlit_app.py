"""Streamlit web UI — K-pop Concert Dynamic Pricing Agent."""
import os
import pathlib
import sys

_HERE = pathlib.Path(__file__).parent.resolve()
os.chdir(_HERE)
sys.path.insert(0, str(_HERE))

from dotenv import load_dotenv
load_dotenv(dotenv_path=_HERE / ".env", override=True)

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import norm

from tools.kopis_tool import list_concerts
from tools.constraint_calibrator import followers_to_popularity


@st.cache_data
def _load_data_stats() -> dict | None:
    """티켓베이 CSV에서 데이터 통계 요약."""
    try:
        df = pd.read_csv(_HERE / "data" / "ticketbay_real.csv")
        price_col = "resale_price" if "resale_price" in df.columns else "listing_price"
        concerts  = df["concert_name"].nunique() if "concert_name" in df.columns else "?"
        return {
            "n_rows":       len(df),
            "n_concerts":   concerts,
            "price_min":    int(df[price_col].min())   if price_col in df.columns else 0,
            "price_max":    int(df[price_col].max())   if price_col in df.columns else 0,
            "price_median": int(df[price_col].median()) if price_col in df.columns else 0,
            "d_day_range":  f"{int(df['d_day'].min())}~{int(df['d_day'].max())}" if "d_day" in df.columns else "?",
        }
    except Exception:
        return None
from tools.simulation import simulate_sensitivity_scenarios
from main import run_agent

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="K-pop Dynamic Pricing Agent",
    page_icon="🎵",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="metric-container"] {
    background: #f8f9fa;
    border: 1px solid #e9ecef;
    border-radius: 10px;
    padding: 12px 16px;
}
.insight-box {
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    border-left: 4px solid #667eea;
    border-radius: 8px;
    padding: 16px 20px;
    margin: 8px 0;
    font-size: 0.9rem;
    line-height: 1.7;
    color: #111111 !important;
}
</style>
""", unsafe_allow_html=True)

st.title("🎵 K-pop Concert Dynamic Pricing Agent")
st.caption(
    "티켓베이 실거래 데이터 기반 WTP 회귀 → LP 최적화 → 동적 가격 책정  ·  "
    "Claude AI 좌석 분석 + KOPIS 실시간 연동"
)

# ── Session state ─────────────────────────────────────────────────────────────
for _k, _v in [("concerts", []), ("result", None), ("searched_artist", "")]:
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🎤 아티스트 입력")
    artist    = st.text_input("아티스트명", placeholder="예: aespa, DAY6, IU")
    followers = st.number_input(
        "Instagram 팔로워 (만명)", min_value=0.1, max_value=10000.0,
        value=150.0, step=1.0, help="150만명 → 150 입력",
    )
    if followers:
        pop = followers_to_popularity(followers)
        label = {7:"🔥 메가", 6:"⭐ 톱", 5:"✨ 인기", 4:"🎵 중견",
                 3:"🎶 신진", 2:"🌱 성장", 1:"🌱 인디"}.get(pop, "")
        st.metric("인기도 점수", f"{pop} / 7", label)

    st.divider()
    if st.button("🔍 KOPIS 공연 검색", type="primary",
                 use_container_width=True, disabled=not artist):
        with st.spinner(f"'{artist}' 검색 중..."):
            st.session_state.concerts       = list_concerts(artist)
            st.session_state.searched_artist = artist
            st.session_state.result          = None

    st.divider()
    st.caption("IE209 생산운영관리 팀 프로젝트")

# ── Main layout ───────────────────────────────────────────────────────────────
col_in, col_out = st.columns([1, 2], gap="large")

# ── 입력 패널 ─────────────────────────────────────────────────────────────────
with col_in:
    st.subheader("📋 공연 정보")

    concert_index = None
    extra         = {}
    manual_mode   = False
    concerts      = st.session_state.concerts

    if concerts:
        st.success(f"{len(concerts)}개 공연 발견")
        opts = ["자동 선택 (가장 가까운 공연)"] + [
            f"[{c['index']}]  {c['date']}  |  {c['name']}  @  {c['venue']}"
            for c in concerts
        ]
        chosen = st.selectbox("공연 선택", opts)
        if chosen != opts[0]:
            concert_index = int(chosen.split("]")[0].replace("[", "").strip())

    elif st.session_state.searched_artist:
        st.warning("KOPIS 결과 없음 → 수동 입력")
        manual_mode = True
        venue  = st.text_input("공연장명", placeholder="예: 올림픽체조경기장")
        d_day  = st.number_input("공연까지 남은 일수", 1, 365, 60)
        n_seat = st.number_input("총 좌석 수", 100, 200000, 10000, 500)
        extra  = {
            "skip_kopis": True, "venue": venue,
            "sale_start_d_day": int(d_day), "total_seats": int(n_seat),
        }
    else:
        st.info("사이드바에서 아티스트명을 입력하고 KOPIS 검색을 눌러주세요.")

    st.divider()
    st.subheader("🗺️ 좌석배치도 (선택)")
    uploaded = st.file_uploader(
        "Claude Vision이 구역별 Z1/Z2/Z3 자동 분석",
        type=["png", "jpg", "jpeg", "webp"],
    )
    img_path = None
    if uploaded:
        (_HERE / "data").mkdir(exist_ok=True)
        img_path = str(_HERE / "data" / uploaded.name)
        with open(img_path, "wb") as f:
            f.write(uploaded.read())
        st.image(img_path, caption="업로드된 좌석배치도", use_column_width=True)

    st.divider()
    disabled = not artist or not followers or (not concerts and not manual_mode)
    if st.button("🚀 가격 최적화 실행", type="primary",
                 use_container_width=True, disabled=disabled):
        inp = {"artist": artist, "followers_in_10k": followers,
               "concert_index": concert_index, **extra}
        if img_path:
            inp["venue_image_path"] = img_path

        with col_out:
            with st.status("AI 에이전트 실행 중...", expanded=True) as status:
                st.write("① 콘서트 정보 확인 + KOPIS 조회")
                st.write("② Claude API 좌석 헤도닉 변수 할당")
                st.write("③ WTP 회귀 분석 → floor/ceiling 도출")
                st.write("④ LP 최적화 (D60/D30/D14/D7/D1)")
                st.write("⑤ 3전략 시뮬레이션 + 민감도 분석 + 시각화")
                try:
                    st.session_state.result = run_agent(inp)
                    status.update(label="✅ 완료!", state="complete")
                except Exception as e:
                    status.update(label=f"❌ 오류: {e}", state="error")
                    st.error(str(e))

# ── 결과 패널 ─────────────────────────────────────────────────────────────────
with col_out:
    if not st.session_state.result:
        st.info("실행 결과가 여기에 표시됩니다.")
        st.stop()

    res  = st.session_state.result
    kpi  = res.get("kpi",            {})
    cons = res.get("constraints",    {})
    lp   = res.get("pricing_result", {})
    wtp  = res.get("wtp_model",      {})
    ci   = res.get("concert_info",   {})
    ins  = res.get("insight",        "")

    total_seats = ci.get("total_seats", 10000)
    floor_p     = float(cons.get("floor",    50000))
    ceiling_p   = float(cons.get("ceiling", 300000))
    mu          = float(wtp.get("mu_final",  150000))
    sigma       = float(wtp.get("sigma",      45000))
    b1m         = float(wtp.get("beta1_over_mu", -0.003))

    # ── KPI 메트릭 ────────────────────────────────────────────────────────────
    st.subheader("📊 핵심 KPI")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Revenue Gain",     f"+{kpi.get('revenue_gain_pct', 0):.1f}%", "동적 vs 고정가")
    m2.metric("MAPE",              f"{kpi.get('mape', 0):.3f}")
    m3.metric("Floor (하한)",  f"{floor_p/10000:.1f}만원")
    m4.metric("Ceiling (상한)", f"{ceiling_p/10000:.1f}만원")

    w1, w2, w3 = st.columns(3)
    w1.metric("μ_final (WTP 평균)", f"{mu/10000:.1f}만원")
    w2.metric("σ (표준편차)",        f"{sigma/10000:.1f}만원")
    w3.metric("LOGO MAPE",          f"{wtp.get('logo_mape', 0):.3f}")

    # 데이터 통계
    stats = _load_data_stats()
    if stats:
        with st.expander("📂 학습 데이터 통계 (티켓베이 크롤링)", expanded=False):
            d1, d2, d3, d4, d5 = st.columns(5)
            d1.metric("총 관측치",    f"{stats['n_rows']:,}건")
            d2.metric("공연 수",      f"{stats['n_concerts']}개")
            d3.metric("최저 재판매가", f"{stats['price_min']//10000}만원")
            d4.metric("최고 재판매가", f"{stats['price_max']//10000}만원")
            d5.metric("D-day 범위",   f"D-{stats['d_day_range']}")

    st.divider()

    # ── 탭 ────────────────────────────────────────────────────────────────────
    tab_price, tab_strat, tab_sens, tab_whatif, tab_insight, tab_model = st.tabs([
        "🎫 가격표",
        "📈 3전략 비교",
        "📉 민감도 시나리오",
        "🎚️ What-if 시뮬레이터",
        "🤖 AI 해설",
        "📐 모델 수식",
    ])

    # ── Tab 1: 가격표 ─────────────────────────────────────────────────────────
    with tab_price:
        st.subheader("D-day별 최적 가격")
        _DDAY_ORDER = ["D60", "D30", "D14", "D7", "D1"]

        rows = []
        for iv in _DDAY_ORDER:
            if iv not in lp:
                continue
            v   = lp[iv]
            row = {
                "D-day":    iv,
                "기준가 (원)": int(v["price"]),
                "예상 판매 (석)": int(v.get("quantity", 0)),
                "구간 수익 (M원)": round(v.get("revenue", 0) / 1e6, 1),
            }
            for zone, zp in v.get("zone_prices", {}).items():
                row[zone] = zp
            rows.append(row)

        df_price = pd.DataFrame(rows)

        def _highlight(row):
            iv = row["D-day"]
            colors = {"D60": "#e8f4fd", "D30": "#d4edda", "D14": "#fff3cd",
                      "D7":  "#fde8e8", "D1":  "#f8d7da"}
            bg = colors.get(iv, "")
            return [f"background-color: {bg}; color: #111111" for _ in row]

        st.dataframe(
            df_price.style.apply(_highlight, axis=1).format({
                "기준가 (원)": "{:,}", "예상 판매 (석)": "{:,}", "구간 수익 (M원)": "{:.1f}",
            }),
            use_container_width=True, hide_index=True,
        )

        # 다운로드 버튼
        dl1, dl2 = st.columns(2)
        dl1.download_button(
            "⬇️ 가격표 CSV 다운로드",
            data=df_price.to_csv(index=False, encoding="utf-8-sig"),
            file_name=f"{ci.get('artist', 'pricing')}_prices.csv",
            mime="text/csv",
            use_container_width=True,
        )
        report_png = _HERE / "results" / "report.png"
        if report_png.exists():
            dl2.download_button(
                "⬇️ 분석 차트 PNG 다운로드",
                data=report_png.read_bytes(),
                file_name=f"{ci.get('artist', 'pricing')}_report.png",
                mime="image/png",
                use_container_width=True,
            )

    # ── Tab 2: 3전략 비교 (Plotly) ────────────────────────────────────────────
    with tab_strat:
        _INTERVALS = ["D60", "D30", "D14", "D7", "D1"]
        _DDAYS     = {"D60": 60, "D30": 30, "D14": 14, "D7": 7, "D1": 1}

        def _d_factor(d_day: int) -> float:
            return max(0.5, min(1.0 + b1m * (14 - d_day), 1.5))

        def _demand(price: float, d_day: int) -> float:
            mu_adj = mu * _d_factor(d_day)
            return total_seats * (1 - norm.cdf(price, mu_adj, sigma))

        st.subheader("전략별 수익 비교")

        rev1 = kpi.get("revenue_strategy1", 0)
        rev2 = kpi.get("revenue_strategy2", 0)
        rev3 = kpi.get("revenue_strategy3", 0)

        fig_bar = go.Figure(data=[
            go.Bar(name="S1 Fixed",        x=["Fixed"],   y=[rev1/1e8],
                   marker_color="#a8d8ea", text=[f"{rev1/1e8:.2f}억"], textposition="outside"),
            go.Bar(name="S2 Optimal Static", x=["Static"],  y=[rev2/1e8],
                   marker_color="#f8b195", text=[f"{rev2/1e8:.2f}억"], textposition="outside"),
            go.Bar(name="S3 Dynamic LP",   x=["Dynamic"], y=[rev3/1e8],
                   marker_color="#f67280", text=[f"{rev3/1e8:.2f}억"], textposition="outside"),
        ])
        fig_bar.update_layout(
            title=f"총 수익 비교  (Dynamic Gain: +{kpi.get('revenue_gain_pct',0):.1f}%)",
            yaxis_title="총 수익 (억원)", showlegend=True, height=350,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        st.subheader("D-day별 가격 궤적")
        fig_line = go.Figure()
        strategy_colors = {"Fixed": "#a8d8ea", "Static": "#f8b195", "Dynamic": "#f67280"}
        for s_name, s_data, s_color in [
            ("Fixed",   {iv: lp[iv] for iv in _INTERVALS if iv in lp}, "#a8d8ea"),
            ("Dynamic", lp, "#f67280"),
        ]:
            ys = [s_data[iv]["price"] / 1000 for iv in _INTERVALS if iv in s_data]
            xs = [iv for iv in _INTERVALS if iv in s_data]
            fig_line.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines+markers", name=s_name,
                line=dict(color=s_color, width=2.5), marker=dict(size=8),
                hovertemplate="%{x}: %{y:.0f}k KRW<extra></extra>",
            ))
        # μ_final 기준선
        fig_line.add_hline(y=mu/1000, line_dash="dot", line_color="#95a5a6",
                           annotation_text=f"μ_final {mu/1000:.0f}k")
        fig_line.update_layout(
            yaxis_title="가격 (천원)", height=350,
            xaxis=dict(categoryorder="array", categoryarray=_INTERVALS),
        )
        st.plotly_chart(fig_line, use_container_width=True)

    # ── Tab 3: 민감도 시나리오 (Plotly) ──────────────────────────────────────
    with tab_sens:
        st.subheader("가격 민감도 시나리오 분석")
        st.caption("D-14 기준 · ceiling 제거 → 이탈이 자연 천장 역할")

        with st.spinner("시나리오 계산 중..."):
            scenarios = simulate_sensitivity_scenarios(
                wtp_model=wtp, total_seats=total_seats,
                floor_price=floor_p, d_day=14,
            )

        fig_sens = go.Figure()
        for sc in scenarios:
            xs = [p / 1000 for p in sc["price_candidates"]]
            ys = [r / 1e8  for r in sc["net_revenues"]]
            fig_sens.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines", name=sc["name"],
                line=dict(color=sc["color"], width=2.2),
                hovertemplate="가격: %{x:.0f}k원<br>순수익: %{y:.2f}억원<extra></extra>",
            ))
            fig_sens.add_trace(go.Scatter(
                x=[sc["optimal_price"] / 1000],
                y=[sc["optimal_net_revenue"] / 1e8],
                mode="markers", showlegend=False,
                marker=dict(color=sc["color"], size=11, symbol="star"),
                hovertemplate=f"{sc['name']}<br>최적가: %{{x:.0f}}k<extra></extra>",
            ))

        # LP 추천가 (D14)
        lp_d14 = lp.get("D14", lp.get("D7", {})).get("price", mu)
        fig_sens.add_vline(
            x=lp_d14 / 1000, line_dash="dash", line_color="#7f8c8d",
            annotation_text=f"LP 추천 {lp_d14/1000:.0f}k",
        )
        fig_sens.add_vline(
            x=mu / 1000, line_dash="dot", line_color="#bdc3c7",
            annotation_text=f"μ_final {mu/1000:.0f}k",
        )
        fig_sens.update_layout(
            xaxis_title="가격 (천원)", yaxis_title="순수익 (억원)",
            height=420, legend=dict(orientation="v", x=1.01),
        )
        st.plotly_chart(fig_sens, use_container_width=True)

        # 시나리오 요약 표
        tbl_data = []
        for sc in scenarios:
            gain = sc["revenue_gain_vs_s1_pct"]
            tbl_data.append({
                "시나리오":     sc["name"],
                "가정":         sc["label"],
                "최적가":       f"{sc['optimal_price']/1000:,.0f}k원",
                "순수익 (억)":  f"{sc['optimal_net_revenue']/1e8:.2f}",
                "vs S1":        f"{'▲' if gain >= 0 else '▼'} {abs(gain):.1f}%",
            })
        df_tbl = pd.DataFrame(tbl_data)

        def _color_gain(val):
            if "▼" in str(val):
                return "color: #c0392b; font-weight: bold"
            return "color: #1a7a3c; font-weight: bold"

        st.dataframe(
            df_tbl.style.map(_color_gain, subset=["vs S1"]),
            use_container_width=True, hide_index=True,
        )

        st.info(
            "**해석 가이드**  \n"
            "★점 = 해당 시나리오의 순수익 최대 가격  |  "
            "LP 추천가(점선)가 S4/S5 ★보다 **오른쪽** = 현실에서 손해  \n"
            "S4·S5는 구매포기 + 브랜드 이미지 훼손 패널티 포함"
        )

        sens_png = _HERE / "results" / "sensitivity_report.png"
        if sens_png.exists():
            st.download_button(
                "⬇️ 시나리오 차트 PNG 다운로드",
                data=sens_png.read_bytes(),
                file_name=f"{ci.get('artist','pricing')}_sensitivity.png",
                mime="image/png",
            )

    # ── Tab 4: What-if 시뮬레이터 ────────────────────────────────────────────
    with tab_whatif:
        st.subheader("🎚️ What-if 가격 시뮬레이터")
        st.caption("슬라이더로 가격을 바꾸면 수요·수익이 실시간으로 바뀝니다.")

        wf_col1, wf_col2 = st.columns([1, 2])

        with wf_col1:
            selected_dday = st.selectbox(
                "D-day 구간 선택",
                ["D60", "D30", "D14", "D7", "D1"],
                index=2,
            )
            d_val = _DDAYS[selected_dday]
            mu_adj_wf = mu * _d_factor(d_val)

            lp_price_wf = int(lp.get(selected_dday, {}).get("price", mu))
            slider_min  = int(floor_p)
            slider_max  = int(mu * 3)

            sel_price = st.slider(
                "가격 설정 (원)",
                min_value=slider_min,
                max_value=slider_max,
                value=lp_price_wf,
                step=5000,
                format="%d원",
            )

            demand_wf  = total_seats * (1 - norm.cdf(sel_price, mu_adj_wf, sigma))
            revenue_wf = sel_price * demand_wf

            st.metric("예상 수요",  f"{int(demand_wf):,}석")
            st.metric("예상 수익",  f"{revenue_wf/1e8:.3f}억원")
            diff = sel_price - lp_price_wf
            st.metric(
                "LP 추천가 대비",
                f"{diff:+,}원",
                delta_color="inverse" if diff > 0 else "normal",
            )

            lp_demand  = total_seats * (1 - norm.cdf(lp_price_wf, mu_adj_wf, sigma))
            lp_revenue = lp_price_wf * lp_demand
            rev_diff_pct = (revenue_wf - lp_revenue) / lp_revenue * 100 if lp_revenue else 0
            st.metric("LP 대비 수익 차이", f"{rev_diff_pct:+.1f}%")

        with wf_col2:
            prices_wf   = np.linspace(slider_min, slider_max, 150)
            revenues_wf = prices_wf * total_seats * (1 - norm.cdf(prices_wf, mu_adj_wf, sigma))
            demands_wf  = total_seats * (1 - norm.cdf(prices_wf, mu_adj_wf, sigma))

            fig_wf = go.Figure()

            # 수익 곡선
            fig_wf.add_trace(go.Scatter(
                x=prices_wf / 1000, y=revenues_wf / 1e8,
                mode="lines", name="수익 곡선",
                line=dict(color="#3498db", width=2.5),
                hovertemplate="가격: %{x:.0f}k원<br>수익: %{y:.3f}억원<extra></extra>",
            ))

            # 선택한 가격 포인트
            fig_wf.add_trace(go.Scatter(
                x=[sel_price / 1000], y=[revenue_wf / 1e8],
                mode="markers", name="선택한 가격",
                marker=dict(color="#e74c3c", size=14, symbol="star"),
                hovertemplate=f"선택가: {sel_price/1000:.0f}k<br>수익: {revenue_wf/1e8:.3f}억<extra></extra>",
            ))

            # LP 최적가 수직선
            fig_wf.add_vline(
                x=lp_price_wf / 1000, line_dash="dash", line_color="#27ae60",
                annotation_text=f"LP 최적 {lp_price_wf/1000:.0f}k",
                annotation_font_color="#27ae60",
            )

            # 수요 곡선 (보조 y축)
            fig_wf.add_trace(go.Scatter(
                x=prices_wf / 1000, y=demands_wf,
                mode="lines", name="수요 (석)",
                line=dict(color="#95a5a6", width=1.5, dash="dot"),
                yaxis="y2",
                hovertemplate="가격: %{x:.0f}k원<br>수요: %{y:.0f}석<extra></extra>",
            ))

            fig_wf.update_layout(
                title=f"{selected_dday} 수익·수요 곡선  (μ_adj = {mu_adj_wf/1000:.0f}k원)",
                xaxis_title="가격 (천원)",
                yaxis=dict(title=dict(text="수익 (억원)", font=dict(color="#3498db"))),
                yaxis2=dict(title=dict(text="수요 (석)", font=dict(color="#95a5a6")),
                            overlaying="y", side="right"),
                height=420,
                legend=dict(orientation="h", y=-0.2),
            )
            st.plotly_chart(fig_wf, use_container_width=True)

            st.caption(
                "📌  빨간 ★ = 현재 선택 가격  |  초록 점선 = LP 추천가 (수익 최대점)  |  "
                "회색 점선 = 수요 곡선 (오른쪽 축)"
            )

    # ── Tab 5: Claude AI 해설 ─────────────────────────────────────────────────
    with tab_insight:
        st.subheader("🤖 AI 전략 해설 (Claude)")
        if ins:
            st.markdown(
                f'<div class="insight-box">{ins.replace(chr(10), "<br>")}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.info("Claude insight가 생성되지 않았습니다. (API 키 확인 필요)")

    # ── Tab 6: 모델 수식 ──────────────────────────────────────────────────────
    with tab_model:
        st.subheader("📐 모델 수식 요약")
        st.caption("본 에이전트에서 사용하는 핵심 수식입니다.")

        st.markdown("#### 1. 헤도닉 좌석 가중치 (Hedonic Seat Weight)")
        st.latex(r"W_g = \beta_{Z1} \cdot Z1_g + \beta_{Z2} \cdot Z2_g + \beta_{Z3} \cdot Z3_g")
        st.markdown("""
- $Z1$: 무대 근접도 (0=최원거리, 1=최근접)
- $Z2$: 정면성 (0=측면, 1=정면)
- $Z3$: 런웨이 인접 여부 (0 or 1)
- $\\beta$ 계수는 티켓베이 구역별 재판매가 OLS 회귀로 추정
        """)

        st.divider()
        st.markdown("#### 2. WTP 평균 조정 (Temporal D-factor)")
        st.latex(r"\mu_{adj,t} = \mu_{final} \times \underbrace{\text{clamp}\!\left(1 + \frac{\beta_1}{\mu_{base}} \cdot (14 - D_t),\ 0.5,\ 1.5\right)}_{d_t}")
        st.markdown("""
- $D_t$: 공연까지 남은 일수 (D60, D30, D14, D7, D1)
- $d_t > 1$: 공연이 가까울수록 팬들의 WTP 상승
- $\\mu_{final} = \\mu_{base} \\times f(\\text{popularity score})$
        """)

        st.divider()
        st.markdown("#### 3. WTP 수요 함수 (Demand Model, B-method)")
        st.latex(r"Q_t(P_t) = N \times \left(1 - \Phi\!\left(\frac{P_t - \mu_{adj,t}}{\sigma}\right)\right)")
        st.markdown("""
- $\\Phi$: 표준 정규 CDF
- $N$: 총 좌석 수
- 가격 $P_t$가 높아질수록 수요 감소 — 정규분포 꼬리 면적
        """)

        st.divider()
        st.markdown("#### 4. LP 최적화 (Price Optimization)")
        st.latex(r"\max_{P_t} \sum_{t \in \mathcal{T}} P_t \cdot Q_t(P_t)")
        st.latex(r"\text{s.t.} \quad P_{floor} \leq P_t \leq P_{ceiling}, \quad \sum_{k} x_{t,k} = 1 \quad (x_{t,k} \in \{0,1\})")
        st.markdown("""
- 비선형 목적함수 → 가격 후보 20개 이산화 후 Binary LP로 선형화
- Solver: PuLP CBC
- $P_{floor}, P_{ceiling}$: WTP 분포에서 자동 도출 ($\\mu_{final} \\pm c \\cdot \\sigma$)
        """)

        st.divider()
        st.markdown("#### 5. 구역별 가격 (Zone Pricing)")
        st.latex(r"P_{t,g} = P_t \times \frac{W_g}{\bar{W}}, \quad \bar{W} = \frac{1}{G}\sum_g W_g")
        st.markdown("기준가 $P_t$에 헤도닉 가중치 비율을 곱해 구역별 차등 가격 산출")

        st.divider()
        st.markdown("#### 6. 가격 민감도 시나리오 (Sensitivity)")
        st.latex(r"\text{churn} = \min\!\left(0.99,\ r \cdot \frac{P - P_{floor}}{0.1 \cdot P_{floor}}\right)")
        st.latex(r"R_{net} = P \cdot Q \cdot (1-\text{churn}) - \underbrace{Q \cdot \text{churn} \cdot \mu_{final} \cdot \alpha}_{\text{brand penalty}}")
        st.markdown("""
| 시나리오 | $r$ (10% 인상당 이탈) | $\\alpha$ (브랜드 패널티) |
|---------|----------------------|------------------------|
| S1 (현재 모델) | 0% | 0% |
| S2 | 3% | 0% |
| S3 | 6% | 0% |
| S4 | 10% | 5% |
| S5 | 15% | 12% |
        """)

    # ── 오류 ──────────────────────────────────────────────────────────────────
    if res.get("errors"):
        with st.expander("⚠️ 실행 중 발생한 경고"):
            for e in res["errors"]:
                st.warning(e)
