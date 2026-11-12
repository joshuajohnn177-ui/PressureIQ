"""PressureIQ: AI/ML Player Pressure Intelligence Dashboard.

Built for the Rajasthan Royals 'SupeRR Selector Season 2' Hackathon.
Interactive Streamlit application for scouting, evaluating, and visualizing
clutch performance in T20 cricket (2022-2026).
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PressureIQ | Rajasthan Royals Analytics",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# PATHS & CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = ROOT_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS (Rajasthan Royals & Professional IPL Aesthetic)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Global Color Variables */
    :root {
        --rr-pink: #ff007f;
        --rr-blue: #001f54;
        --rr-navy: #0a1128;
        --rr-gold: #ffd700;
        --rr-card: #132043;
        --rr-accent: #00d4ff;
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }

    /* Hero Banner */
    .hero-container {
        background: linear-gradient(135deg, #0a1128 0%, #001f54 60%, #1e0034 100%);
        border: 1px solid rgba(255, 0, 127, 0.3);
        border-radius: 14px;
        padding: 1.75rem 2rem;
        margin-bottom: 1.75rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.45);
    }
    .hero-title {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #ff007f, #ffd700, #00d4ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.25rem;
        letter-spacing: -0.5px;
    }
    .hero-subtitle {
        color: #e0e6ed;
        font-size: 1.05rem;
        font-weight: 400;
        margin-bottom: 0.5rem;
    }
    .hero-badges {
        display: flex;
        gap: 0.6rem;
        flex-wrap: wrap;
        margin-top: 0.75rem;
    }
    .badge {
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.15);
        color: #f8f9fa;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-pink {
        background: rgba(255, 0, 127, 0.2);
        border-color: #ff007f;
        color: #ff66b2;
    }
    .badge-gold {
        background: rgba(255, 215, 0, 0.2);
        border-color: #ffd700;
        color: #ffe066;
    }
    .badge-blue {
        background: rgba(0, 212, 255, 0.2);
        border-color: #00d4ff;
        color: #66e5ff;
    }

    /* Metric Cards */
    .metric-card {
        background: linear-gradient(145deg, #132043 0%, #0d1730 100%);
        border: 1px solid rgba(255, 255, 255, 0.09);
        border-radius: 12px;
        padding: 1.25rem 1.25rem 1rem 1.25rem;
        margin-bottom: 1rem;
        transition: transform 0.2s ease, border-color 0.2s ease;
        box-shadow: 0 4px 16px rgba(0,0,0,0.25);
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(255, 0, 127, 0.4);
    }
    .metric-title {
        color: #94a3b8;
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.35rem;
    }
    .metric-val {
        color: #ffffff;
        font-size: 1.85rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }
    .metric-delta-pos {
        color: #10b981;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .metric-delta-neg {
        color: #ef4444;
        font-size: 0.85rem;
        font-weight: 600;
    }

    /* Leader Rank Badges */
    .rank-1 {
        background: linear-gradient(135deg, #ffd700, #ffae00);
        color: #000;
        font-weight: 800;
        padding: 4px 12px;
        border-radius: 8px;
    }
    .rank-2 {
        background: linear-gradient(135deg, #e2e8f0, #94a3b8);
        color: #000;
        font-weight: 800;
        padding: 4px 12px;
        border-radius: 8px;
    }
    .rank-3 {
        background: linear-gradient(135deg, #cd7f32, #a05a2c);
        color: #fff;
        font-weight: 800;
        padding: 4px 12px;
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADERS (Cached for Speed)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_top10():
    """Load top 10 official submission rankings."""
    bat_path = OUTPUTS_DIR / "top10_batters.csv"
    bowl_path = OUTPUTS_DIR / "top10_bowlers.csv"
    df_bat = pd.read_csv(bat_path)
    df_bowl = pd.read_csv(bowl_path)
    return df_bat, df_bowl


@st.cache_data(show_spinner=False)
def load_all_players():
    """Load complete eligible batter and bowler analytics."""
    bat_path = OUTPUTS_DIR / "all_batters_pressure_metrics.parquet"
    bowl_path = OUTPUTS_DIR / "all_bowlers_pressure_metrics.parquet"
    df_bat = pd.read_parquet(bat_path)
    df_bowl = pd.read_parquet(bowl_path)
    return df_bat, df_bowl


@st.cache_data(show_spinner=False)
def load_match_deliveries():
    """Load subset of delivery-level WPA data for match analysis."""
    cols = [
        "match_id",
        "innings",
        "overid",
        "over_ball_num",
        "innings_ball_num",
        "home_team",
        "away_team",
        "season",
        "competition_tier",
        "striker_name",
        "bowler_name",
        "runs",
        "ball_total_runs",
        "is_wicket",
        "win_prob_pre",
        "win_prob_post",
        "wpa",
        "batter_wpa",
        "bowler_wpa",
        "leverage_index",
        "is_pressure",
        "crr_pre",
        "rrr_pre",
    ]
    df = pd.read_parquet(OUTPUTS_DIR / "deliveries_with_wpa.parquet", columns=cols)
    return df


@st.cache_data(show_spinner=False)
def load_model_metrics():
    """Load model evaluation comparison."""
    with open(OUTPUTS_DIR / "model_evaluation_metrics.json", "r") as f:
        metrics = json.load(f)
    return pd.DataFrame(metrics)


# Load data assets
df_top_bat, df_top_bowl = load_top10()
df_all_bat, df_all_bowl = load_all_players()


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR NAVIGATION & CONTROLS
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏏 **PressureIQ**")
    st.markdown("*Rajasthan Royals SupeRR Selector*")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "🏆 Official Top 10 Leaderboards",
            "🔍 Player Scouting & Deep-Dive",
            "⚔️ Head-to-Head Comparison",
            "📊 Situational Differentials (Matrix)",
            "⚡ Match Win Probability & WPA Dynamics",
            "📖 Methodology & Validation Engine",
        ],
        index=0,
    )

    st.markdown("---")
    st.markdown("### 🎛️ **Scouting Filters**")

    min_balls_filter = st.slider(
        "Min. Pressure Balls",
        min_value=50,
        max_value=300,
        value=50,
        step=25,
        help="Filter players by high-pressure delivery sample size.",
    )

    st.markdown("---")
    st.markdown("### 📈 **Dataset Statistics**")
    st.markdown(
        f"""
        - **Total Deliveries:** `199,029`
        - **Historical Matches:** `842` (2022–2026)
        - **Pressure Deliveries:** `67,290` (33.8%)
        - **Eligible Batters:** `{len(df_all_bat)}`
        - **Eligible Bowlers:** `{len(df_all_bowl)}`
        """
    )
    st.markdown("---")
    st.caption("Developed for RR SupeRR Selector 2026. All ML models deterministic.")


# ─────────────────────────────────────────────────────────────────────────────
# HERO BANNER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero-container">
        <div class="hero-title">PressureIQ: Player Intelligence Engine</div>
        <div class="hero-subtitle">
            Calibrated Machine Learning Win Probability, Dynamic Leverage Index, and Empirical Bayes Regularization for T20 Cricket.
        </div>
        <div class="hero-badges">
            <span class="badge badge-pink">Rajasthan Royals Hackathon</span>
            <span class="badge badge-gold">Dual-Innings Ensemble Model (ROC-AUC 0.816)</span>
            <span class="badge badge-blue">Zero Data Leakage Audit Verified</span>
            <span class="badge">28/28 Tests Passing</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1: OFFICIAL TOP 10 LEADERBOARDS
# ─────────────────────────────────────────────────────────────────────────────
if page == "🏆 Official Top 10 Leaderboards":
    st.markdown("### 🏆 PressureIQ Top 10 – Model Rankings")
    st.markdown(
        "Composite rankings are built from calibrated ML Win Probability Added (WPA), "
        "situational performance differentials, and Empirical Bayes reliability weighting — "
        "applied across **67,290 high-pressure deliveries** from 2022–2026."
    )
    with st.expander("ℹ️ How is the PressureIQ Score calculated?"):
        st.markdown(
            """
            The **PressureIQ Composite Score** combines three independently standardised components:

            | Component | Weight | Description |
            |:---|:---:|:---|
            | **Shrunk Total Pressure WPA** | **60%** | Z-score of cumulative Win Probability Added under high-pressure deliveries, after Empirical Bayes shrinkage (λᵢ × WPA/ball × n). Rewards consistent match-winning impact. |
            | **Situational Differential** | **25%** | Z-score of shrunk Strike Rate uplift (batters) or Economy Rate suppression (bowlers) under pressure vs their own baseline. |
            | **Empirical Bayes Credibility** | **15%** | Raw credibility weight λᵢ = nᵢ / (nᵢ + 75), reflecting sample reliability. Players with ≥ 500 pressure balls earn λ ≥ 0.87; minimum threshold is 50 balls. |

            **Formula:**
            - Batters: `PressureIQ = 60·Z(Shrunk WPA) + 25·Z(ΔSR_shrunk) + 15·λᵢ`
            - Bowlers: `PressureIQ = 60·Z(Shrunk WPA) + 25·Z(-ΔEcon_shrunk) + 15·λᵢ`

            > Weights verified against `src/player_analytics.py` lines 113 and 214.
            """
        )
    st.markdown(
        "Identifying the most reliable players under game-defining high-pressure situations across IPL and T20Is (2022–2026)."
    )

    # Top Highlight Cards
    col_lead1, col_lead2 = st.columns(2)

    with col_lead1:
        st.markdown(
            """
            <div class="metric-card" style="border-left: 4px solid #ff007f;">
                <div class="metric-title">🏏 #1 Ranked Batter Under Pressure</div>
                <div class="metric-val" style="color: #ff007f;">Suryakumar Yadav</div>
                <div style="display: flex; justify-content: space-between; margin-top: 0.5rem;">
                    <div><b>PressureIQ Score:</b> 403.24</div>
                    <div class="metric-delta-pos"><b>SR Diff:</b> +24.96</div>
                    <div><b>Pressure WPA:</b> +3.35</div>
                </div>
                <div style="font-size: 0.85rem; color: #94a3b8; margin-top: 0.35rem;">
                    59 matches · 827 pressure balls · Pressure SR: 191.41
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_lead2:
        st.markdown(
            """
            <div class="metric-card" style="border-left: 4px solid #ffd700;">
                <div class="metric-title">🎯 #1 Ranked Bowler Under Pressure</div>
                <div class="metric-val" style="color: #ffd700;">Rashid Khan</div>
                <div style="display: flex; justify-content: space-between; margin-top: 0.5rem;">
                    <div><b>PressureIQ Score:</b> 301.18</div>
                    <div><b>Pressure Wickets:</b> 84</div>
                    <div class="metric-delta-pos"><b>Pressure WPA:</b> +2.60</div>
                </div>
                <div style="font-size: 0.85rem; color: #94a3b8; margin-top: 0.35rem;">
                    91 matches · 955 pressure balls · Pressure Econ: 8.12
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Leaderboard Tabs
    tab_bat, tab_bowl = st.tabs(["🏏 Top 10 Batters", "🎯 Top 10 Bowlers"])

    with tab_bat:
        # Dynamic filter display
        filtered_batters = df_all_bat[df_all_bat["pressure_balls"] >= min_balls_filter].head(10)

        # Plotly Horizontal Bar of Scores
        fig_bat = px.bar(
            df_top_bat.sort_values("PressureIQ Score", ascending=True),
            x="PressureIQ Score",
            y="Player Name",
            orientation="h",
            text="PressureIQ Score",
            color="Pressure SR",
            color_continuous_scale="Purples",
            title="Top 10 Batters by Composite PressureIQ Score",
        )
        fig_bat.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#f8f9fa",
            height=400,
            margin=dict(l=10, r=10, t=40, b=10),
            xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)"),
            yaxis=dict(showgrid=False),
        )
        st.plotly_chart(fig_bat, width="stretch")

        # Formatted Data Table
        st.markdown("#### Official Submission Table (Batters)")
        st.dataframe(
            df_top_bat.style.format(
                {
                    "Baseline SR": "{:.2f}",
                    "Pressure SR": "{:.2f}",
                    "SR Differential (+/-)": "{:+.2f}",
                    "Total Pressure WPA": "{:+.4f}",
                    "Shrunk WPA / Ball": "{:.5f}",
                    "PressureIQ Score": "{:.2f}",
                    "Reliability Weight": "{:.3f}",
                }
            ).background_gradient(subset=["PressureIQ Score"], cmap="plasma"),
            width="stretch",
            hide_index=True,
        )

        csv_bat = df_top_bat.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download top10_batters.csv",
            csv_bat,
            "top10_batters.csv",
            "text/csv",
            key="dl-bat",
        )

    with tab_bowl:
        fig_bowl = px.bar(
            df_top_bowl.sort_values("PressureIQ Score", ascending=True),
            x="PressureIQ Score",
            y="Player Name",
            orientation="h",
            text="PressureIQ Score",
            color="Total Pressure WPA",
            color_continuous_scale="Viridis",
            title="Top 10 Bowlers by Composite PressureIQ Score",
        )
        fig_bowl.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#f8f9fa",
            height=400,
            margin=dict(l=10, r=10, t=40, b=10),
            xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)"),
            yaxis=dict(showgrid=False),
        )
        st.plotly_chart(fig_bowl, width="stretch")

        st.markdown("#### Official Submission Table (Bowlers)")
        st.dataframe(
            df_top_bowl.style.format(
                {
                    "Baseline Economy": "{:.2f}",
                    "Pressure Economy": "{:.2f}",
                    "Economy Differential (+/-)": "{:+.2f}",
                    "Total Pressure WPA": "{:+.4f}",
                    "Shrunk WPA / Ball": "{:.5f}",
                    "PressureIQ Score": "{:.2f}",
                    "Reliability Weight": "{:.3f}",
                }
            ).background_gradient(subset=["PressureIQ Score"], cmap="viridis"),
            width="stretch",
            hide_index=True,
        )

        csv_bowl = df_top_bowl.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download top10_bowlers.csv",
            csv_bowl,
            "top10_bowlers.csv",
            "text/csv",
            key="dl-bowl",
        )


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2: PLAYER SCOUTING & DEEP-DIVE
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🔍 Player Scouting & Deep-Dive":
    st.markdown("### 🔍 Player Scouting & Deep-Dive Analytics")
    st.markdown("Explore comprehensive situational pressure breakdowns across all 536 eligible T20 players.")

    role_choice = st.radio("Select Role", ["🏏 Batter", "🎯 Bowler"], horizontal=True)

    if role_choice == "🏏 Batter":
        player_list = df_all_bat["striker_name"].tolist()
        default_idx = player_list.index("Suryakumar Yadav") if "Suryakumar Yadav" in player_list else 0
        selected_player = st.selectbox("Search & Select Batter", player_list, index=default_idx)

        pdata = df_all_bat[df_all_bat["striker_name"] == selected_player].iloc[0]

        # Profile Card
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Rank & Score</div>
                    <div class="metric-val">#{int(pdata['rank'])}</div>
                    <div class="metric-delta-pos">Score: {pdata['pressure_iq_score']:.2f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c2:
            sr_diff = pdata["sr_differential"]
            diff_class = "metric-delta-pos" if sr_diff >= 0 else "metric-delta-neg"
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Strike Rate (Pressure vs Base)</div>
                    <div class="metric-val">{pdata['pressure_sr']:.1f}</div>
                    <div class="{diff_class}">Diff: {sr_diff:+.1f} (Base: {pdata['overall_sr']:.1f})</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Total Pressure WPA</div>
                    <div class="metric-val">{pdata['pressure_wpa']:+.3f}</div>
                    <div style="font-size: 0.85rem; color: #94a3b8;">Per Ball: {pdata['pressure_wpa_per_ball']:.5f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c4:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Credibility & Sample</div>
                    <div class="metric-val">{pdata['sample_credibility']:.2f}</div>
                    <div style="font-size: 0.85rem; color: #94a3b8;">{int(pdata['pressure_balls'])} balls · {int(pdata['pressure_matches'])} matches</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Radar / Attributes Comparison
        col_rad, col_bar = st.columns(2)

        with col_rad:
            categories = ["Strike Rate", "Boundary %", "Dot Ball %", "WPA/100 Balls", "Leverage Faced"]
            # Normalize to 0-100 scale for intuitive radar comparison
            player_vals = [
                min(100, (pdata["pressure_sr"] / 200.0) * 100),
                min(100, (pdata["pressure_boundary_pct"] / 30.0) * 100),
                max(0, 100 - (pdata["pressure_dot_pct"] / 50.0) * 100),  # Invert dots: fewer dots = better
                min(100, max(0, (pdata["pressure_wpa_per_ball"] * 100 + 0.5) * 50)),
                min(100, (pdata["avg_leverage"] / 2.0) * 100),
            ]
            baseline_vals = [
                min(100, (pdata["overall_sr"] / 200.0) * 100),
                min(100, (pdata["overall_boundary_pct"] / 30.0) * 100),
                max(0, 100 - (pdata["overall_dot_pct"] / 50.0) * 100),
                50,
                50,
            ]

            fig_radar = go.Figure()
            fig_radar.add_trace(
                go.Scatterpolar(
                    r=player_vals + [player_vals[0]],
                    theta=categories + [categories[0]],
                    fill="toself",
                    fillcolor="rgba(255, 0, 127, 0.25)",
                    line=dict(color="#ff007f", width=2),
                    name="High Pressure",
                )
            )
            fig_radar.add_trace(
                go.Scatterpolar(
                    r=baseline_vals + [baseline_vals[0]],
                    theta=categories + [categories[0]],
                    fill="toself",
                    fillcolor="rgba(0, 212, 255, 0.15)",
                    line=dict(color="#00d4ff", width=2, dash="dash"),
                    name="Overall Baseline",
                )
            )
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 100], showticklabels=False),
                    bgcolor="rgba(0,0,0,0)",
                ),
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#f8f9fa",
                title=f"{selected_player}: Situational Radar",
                height=380,
                margin=dict(l=40, r=40, t=40, b=20),
            )
            st.plotly_chart(fig_radar, width="stretch")

        with col_bar:
            # Metric Comparison Bar
            comp_df = pd.DataFrame(
                {
                    "Metric": ["Strike Rate", "Boundary %", "Dot Ball %"],
                    "Baseline": [pdata["overall_sr"], pdata["overall_boundary_pct"], pdata["overall_dot_pct"]],
                    "Under Pressure": [
                        pdata["pressure_sr"],
                        pdata["pressure_boundary_pct"],
                        pdata["pressure_dot_pct"],
                    ],
                }
            )
            comp_melted = pd.melt(comp_df, id_vars=["Metric"], value_vars=["Baseline", "Under Pressure"])
            fig_comp = px.bar(
                comp_melted,
                x="Metric",
                y="value",
                color="variable",
                barmode="group",
                color_discrete_map={"Baseline": "#64748b", "Under Pressure": "#ff007f"},
                title=f"{selected_player}: Baseline vs High-Pressure Stats",
            )
            fig_comp.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#f8f9fa",
                height=380,
                margin=dict(l=10, r=10, t=40, b=20),
            )
            st.plotly_chart(fig_comp, width="stretch")

    else:
        player_list = df_all_bowl["bowler_name"].tolist()
        default_idx = player_list.index("Rashid Khan") if "Rashid Khan" in player_list else 0
        selected_player = st.selectbox("Search & Select Bowler", player_list, index=default_idx)

        pdata = df_all_bowl[df_all_bowl["bowler_name"] == selected_player].iloc[0]

        # Profile Card
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Rank & Score</div>
                    <div class="metric-val">#{int(pdata['rank'])}</div>
                    <div class="metric-delta-pos">Score: {pdata['pressure_iq_score']:.2f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c2:
            econ_diff = pdata["econ_differential"]
            diff_class = "metric-delta-pos" if econ_diff <= 0 else "metric-delta-neg"
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Economy (Pressure vs Base)</div>
                    <div class="metric-val">{pdata['pressure_econ']:.2f}</div>
                    <div class="{diff_class}">Diff: {econ_diff:+.2f} (Base: {pdata['overall_econ']:.2f})</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Pressure Wickets & WPA</div>
                    <div class="metric-val">{int(pdata['pressure_wickets'])} Wkts</div>
                    <div class="metric-delta-pos">WPA: {pdata['pressure_wpa']:+.3f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c4:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Credibility & Sample</div>
                    <div class="metric-val">{pdata['sample_credibility']:.2f}</div>
                    <div style="font-size: 0.85rem; color: #94a3b8;">{int(pdata['pressure_balls'])} balls · {int(pdata['pressure_matches'])} matches</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        col_rad, col_bar = st.columns(2)

        with col_rad:
            categories = ["Economy Suppression", "Dot Ball %", "Boundary Restriction", "WPA/100 Balls", "Leverage"]
            player_vals = [
                max(0, min(100, 100 - (pdata["pressure_econ"] / 12.0) * 100)),
                min(100, (pdata["pressure_dot_pct"] / 60.0) * 100),
                max(0, 100 - (pdata["pressure_boundary_pct"] / 30.0) * 100),
                min(100, max(0, (pdata["pressure_wpa_per_ball"] * 100 + 0.5) * 50)),
                min(100, (pdata["avg_leverage"] / 2.0) * 100),
            ]
            baseline_vals = [
                max(0, min(100, 100 - (pdata["overall_econ"] / 12.0) * 100)),
                min(100, (pdata["overall_dot_pct"] / 60.0) * 100),
                max(0, 100 - (pdata["overall_boundary_pct"] / 30.0) * 100),
                50,
                50,
            ]

            fig_radar = go.Figure()
            fig_radar.add_trace(
                go.Scatterpolar(
                    r=player_vals + [player_vals[0]],
                    theta=categories + [categories[0]],
                    fill="toself",
                    fillcolor="rgba(255, 215, 0, 0.25)",
                    line=dict(color="#ffd700", width=2),
                    name="High Pressure",
                )
            )
            fig_radar.add_trace(
                go.Scatterpolar(
                    r=baseline_vals + [baseline_vals[0]],
                    theta=categories + [categories[0]],
                    fill="toself",
                    fillcolor="rgba(0, 212, 255, 0.15)",
                    line=dict(color="#00d4ff", width=2, dash="dash"),
                    name="Overall Baseline",
                )
            )
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 100], showticklabels=False),
                    bgcolor="rgba(0,0,0,0)",
                ),
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#f8f9fa",
                title=f"{selected_player}: Situational Radar",
                height=380,
                margin=dict(l=40, r=40, t=40, b=20),
            )
            st.plotly_chart(fig_radar, width="stretch")

        with col_bar:
            comp_df = pd.DataFrame(
                {
                    "Metric": ["Economy Rate", "Dot Ball %", "Boundary %"],
                    "Baseline": [pdata["overall_econ"], pdata["overall_dot_pct"], pdata["overall_boundary_pct"]],
                    "Under Pressure": [
                        pdata["pressure_econ"],
                        pdata["pressure_dot_pct"],
                        pdata["pressure_boundary_pct"],
                    ],
                }
            )
            comp_melted = pd.melt(comp_df, id_vars=["Metric"], value_vars=["Baseline", "Under Pressure"])
            fig_comp = px.bar(
                comp_melted,
                x="Metric",
                y="value",
                color="variable",
                barmode="group",
                color_discrete_map={"Baseline": "#64748b", "Under Pressure": "#ffd700"},
                title=f"{selected_player}: Baseline vs High-Pressure Stats",
            )
            fig_comp.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#f8f9fa",
                height=380,
                margin=dict(l=10, r=10, t=40, b=20),
            )
            st.plotly_chart(fig_comp, width="stretch")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3: HEAD-TO-HEAD COMPARISON
# ─────────────────────────────────────────────────────────────────────────────
elif page == "⚔️ Head-to-Head Comparison":
    st.markdown("### ⚔️ Head-to-Head Player Comparison")
    st.markdown("Directly evaluate two players side-by-side across pressure metrics, differentials, and reliability.")

    comp_role = st.radio("Select Role", ["🏏 Compare Batters", "🎯 Compare Bowlers"], horizontal=True)

    if comp_role == "🏏 Compare Batters":
        batters_list = df_all_bat["striker_name"].tolist()
        c1, c2 = st.columns(2)
        with c1:
            p1_name = st.selectbox("Select Player 1", batters_list, index=0)
        with c2:
            default_p2 = 1 if len(batters_list) > 1 else 0
            p2_name = st.selectbox("Select Player 2", batters_list, index=default_p2)

        p1 = df_all_bat[df_all_bat["striker_name"] == p1_name].iloc[0]
        p2 = df_all_bat[df_all_bat["striker_name"] == p2_name].iloc[0]

        # Comparative Table
        metrics_comp = [
            ("PressureIQ Score", p1["pressure_iq_score"], p2["pressure_iq_score"], "{:.2f}", True),
            ("Official Rank", p1["rank"], p2["rank"], "#{:.0f}", False),
            ("Pressure SR", p1["pressure_sr"], p2["pressure_sr"], "{:.2f}", True),
            ("Baseline SR", p1["overall_sr"], p2["overall_sr"], "{:.2f}", True),
            ("SR Differential (+/-)", p1["sr_differential"], p2["sr_differential"], "{:+.2f}", True),
            ("Total Pressure WPA", p1["pressure_wpa"], p2["pressure_wpa"], "{:+.4f}", True),
            ("Shrunk WPA / Ball", p1["shrunk_wpa_per_ball"], p2["shrunk_wpa_per_ball"], "{:.5f}", True),
            ("Reliability Weight (λ)", p1["sample_credibility"], p2["sample_credibility"], "{:.3f}", True),
            ("Pressure Deliveries", p1["pressure_balls"], p2["pressure_balls"], "{:.0f}", True),
            ("Boundary %", p1["pressure_boundary_pct"], p2["pressure_boundary_pct"], "{:.1f}%", True),
            ("Dot Ball %", p1["pressure_dot_pct"], p2["pressure_dot_pct"], "{:.1f}%", False),
        ]

        table_rows = []
        for label, v1, v2, fmt, higher_is_better in metrics_comp:
            f1 = fmt.format(v1)
            f2 = fmt.format(v2)
            if higher_is_better:
                adv = p1_name if v1 > v2 else (p2_name if v2 > v1 else "Tie")
            else:
                adv = p1_name if v1 < v2 else (p2_name if v2 < v1 else "Tie")
            table_rows.append({"Metric": label, p1_name: f1, p2_name: f2, "Advantage": adv})

        st.dataframe(pd.DataFrame(table_rows), width="stretch", hide_index=True)

        # Comparative Grouped Bar
        bar_df = pd.DataFrame(
            {
                "Metric": ["Pressure SR", "SR Diff", "Pressure WPA × 10", "Score / 4"],
                p1_name: [
                    p1["pressure_sr"],
                    p1["sr_differential"],
                    p1["pressure_wpa"] * 10,
                    p1["pressure_iq_score"] / 4,
                ],
                p2_name: [
                    p2["pressure_sr"],
                    p2["sr_differential"],
                    p2["pressure_wpa"] * 10,
                    p2["pressure_iq_score"] / 4,
                ],
            }
        )
        bar_melted = pd.melt(bar_df, id_vars=["Metric"], value_vars=[p1_name, p2_name])

        fig_h2h = px.bar(
            bar_melted,
            x="Metric",
            y="value",
            color="variable",
            barmode="group",
            color_discrete_map={p1_name: "#ff007f", p2_name: "#00d4ff"},
            title=f"{p1_name} vs {p2_name}: Key Metrics Comparison",
        )
        fig_h2h.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#f8f9fa",
            height=380,
            margin=dict(l=10, r=10, t=40, b=20),
        )
        st.plotly_chart(fig_h2h, width="stretch")

    else:
        bowler_list = df_all_bowl["bowler_name"].tolist()
        c1, c2 = st.columns(2)
        with c1:
            p1_name = st.selectbox("Select Bowler 1", bowler_list, index=0)
        with c2:
            default_p2 = 1 if len(bowler_list) > 1 else 0
            p2_name = st.selectbox("Select Bowler 2", bowler_list, index=default_p2)

        p1 = df_all_bowl[df_all_bowl["bowler_name"] == p1_name].iloc[0]
        p2 = df_all_bowl[df_all_bowl["bowler_name"] == p2_name].iloc[0]

        metrics_comp = [
            ("PressureIQ Score", p1["pressure_iq_score"], p2["pressure_iq_score"], "{:.2f}", True),
            ("Official Rank", p1["rank"], p2["rank"], "#{:.0f}", False),
            ("Pressure Economy", p1["pressure_econ"], p2["pressure_econ"], "{:.2f}", False),
            ("Baseline Economy", p1["overall_econ"], p2["overall_econ"], "{:.2f}", False),
            ("Econ Differential (+/-)", p1["econ_differential"], p2["econ_differential"], "{:+.2f}", False),
            ("Total Pressure WPA", p1["pressure_wpa"], p2["pressure_wpa"], "{:+.4f}", True),
            ("Pressure Wickets", p1["pressure_wickets"], p2["pressure_wickets"], "{:.0f}", True),
            ("Reliability Weight (λ)", p1["sample_credibility"], p2["sample_credibility"], "{:.3f}", True),
            ("Pressure Deliveries", p1["pressure_balls"], p2["pressure_balls"], "{:.0f}", True),
            ("Dot Ball %", p1["pressure_dot_pct"], p2["pressure_dot_pct"], "{:.1f}%", True),
        ]

        table_rows = []
        for label, v1, v2, fmt, higher_is_better in metrics_comp:
            f1 = fmt.format(v1)
            f2 = fmt.format(v2)
            if higher_is_better:
                adv = p1_name if v1 > v2 else (p2_name if v2 > v1 else "Tie")
            else:
                adv = p1_name if v1 < v2 else (p2_name if v2 < v1 else "Tie")
            table_rows.append({"Metric": label, p1_name: f1, p2_name: f2, "Advantage": adv})

        st.dataframe(pd.DataFrame(table_rows), width="stretch", hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4: SITUATIONAL DIFFERENTIALS (MATRIX)
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📊 Situational Differentials (Matrix)":
    st.markdown("### 📊 Situational Performance Differentials Matrix")
    st.markdown(
        "Mapping baseline ability against high-pressure execution reveals players who genuinely accelerate under pressure versus those who succumb."
    )

    tab_m_bat, tab_m_bowl = st.tabs(["🏏 Batting Differentials Matrix", "🎯 Bowling Differentials Matrix"])

    with tab_m_bat:
        fig_scatter_bat = px.scatter(
            df_all_bat,
            x="overall_sr",
            y="pressure_sr",
            size="pressure_balls",
            color="pressure_iq_score",
            hover_name="striker_name",
            hover_data={
                "overall_sr": ":.2f",
                "pressure_sr": ":.2f",
                "sr_differential": ":+.2f",
                "pressure_wpa": ":+.3f",
                "pressure_balls": True,
            },
            color_continuous_scale="Turbo",
            title="Baseline Strike Rate vs High-Pressure Strike Rate (Batters)",
            labels={
                "overall_sr": "Baseline Strike Rate (Overall)",
                "pressure_sr": "Pressure Strike Rate",
                "pressure_iq_score": "PressureIQ",
            },
        )
        # Add 1:1 diagonal line
        fig_scatter_bat.add_shape(
            type="line",
            x0=80,
            y0=80,
            x1=240,
            y1=240,
            line=dict(color="rgba(255,255,255,0.4)", dash="dash", width=1.5),
        )
        fig_scatter_bat.add_annotation(
            x=130, y=180, text="▲ Clutch Accelerators (SR Jump)", showarrow=False, font=dict(color="#10b981", size=12)
        )
        fig_scatter_bat.add_annotation(
            x=180, y=120, text="▼ Pressure Decelerators", showarrow=False, font=dict(color="#ef4444", size=12)
        )
        fig_scatter_bat.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#f8f9fa",
            height=500,
            margin=dict(l=10, r=10, t=40, b=20),
        )
        st.plotly_chart(fig_scatter_bat, width="stretch")

    with tab_m_bowl:
        fig_scatter_bowl = px.scatter(
            df_all_bowl,
            x="overall_econ",
            y="pressure_econ",
            size="pressure_balls",
            color="pressure_iq_score",
            hover_name="bowler_name",
            hover_data={
                "overall_econ": ":.2f",
                "pressure_econ": ":.2f",
                "econ_differential": ":+.2f",
                "pressure_wpa": ":+.3f",
                "pressure_wickets": True,
            },
            color_continuous_scale="Plasma",
            title="Baseline Economy vs High-Pressure Economy (Bowlers)",
            labels={
                "overall_econ": "Baseline Economy Rate (Overall)",
                "pressure_econ": "Pressure Economy Rate",
                "pressure_iq_score": "PressureIQ",
            },
        )
        fig_scatter_bowl.add_shape(
            type="line",
            x0=5,
            y0=5,
            x1=13,
            y1=13,
            line=dict(color="rgba(255,255,255,0.4)", dash="dash", width=1.5),
        )
        fig_scatter_bowl.add_annotation(
            x=7, y=11, text="▲ Death Inflation / Blowouts", showarrow=False, font=dict(color="#ef4444", size=12)
        )
        fig_scatter_bowl.add_annotation(
            x=10,
            y=6.5,
            text="▼ Elite Run Suppressors (Econ Drop)",
            showarrow=False,
            font=dict(color="#10b981", size=12),
        )
        fig_scatter_bowl.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#f8f9fa",
            height=500,
            margin=dict(l=10, r=10, t=40, b=20),
        )
        st.plotly_chart(fig_scatter_bowl, width="stretch")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 5: MATCH WIN PROBABILITY & WPA DYNAMICS
# ─────────────────────────────────────────────────────────────────────────────
elif page == "⚡ Match Win Probability & WPA Dynamics":
    st.markdown("### ⚡ Ball-by-Ball Match Win Probability Explorer")
    st.markdown("Inspect how our Calibrated ML Model and WPA Engine reconstruct game-defining moments in real time.")

    with st.spinner("Loading match delivery database..."):
        df_deliveries = load_match_deliveries()

    # Pre-selected high-leverage matches
    sample_matches = (
        df_deliveries.groupby(["match_id", "home_team", "away_team", "competition_tier", "season"])[
            "leverage_index"
        ]
        .mean()
        .reset_index()
        .sort_values("leverage_index", ascending=False)
        .head(40)
    )

    match_options = {
        f"Match #{row['match_id']} | {row['home_team']} vs {row['away_team']} ({row['competition_tier']} {row['season']})": row[
            "match_id"
        ]
        for _, row in sample_matches.iterrows()
    }

    selected_match_label = st.selectbox("Select Match to Inspect", list(match_options.keys()))
    selected_match_id = match_options[selected_match_label]

    match_df = (
        df_deliveries[df_deliveries["match_id"] == selected_match_id]
        .sort_values(["innings", "innings_ball_num"])
        .reset_index(drop=True)
    )

    # Win Probability Progression Plot
    match_df["global_ball_idx"] = match_df.index + 1

    fig_wp = go.Figure()
    fig_wp.add_trace(
        go.Scatter(
            x=match_df["global_ball_idx"],
            y=match_df["win_prob_post"],
            mode="lines",
            line=dict(color="#00d4ff", width=2.5),
            name="Batting Team Win Probability",
            hovertemplate="Ball %{x}<br>Win Prob: %{y:.1%}<extra></extra>",
        )
    )

    # Highlight High Leverage Deliveries (LI >= 1.5)
    high_li = match_df[match_df["leverage_index"] >= 1.5]
    if len(high_li) > 0:
        fig_wp.add_trace(
            go.Scatter(
                x=high_li["global_ball_idx"],
                y=high_li["win_prob_post"],
                mode="markers",
                marker=dict(size=7, color="#ff007f", symbol="circle"),
                name="Clutch Moment (LI ≥ 1.5)",
                hovertemplate="Ball %{x}<br>Leverage Index: %{text:.2f}<extra></extra>",
                text=high_li["leverage_index"],
            )
        )

    # Add 50% line
    fig_wp.add_hline(y=0.5, line_dash="dash", line_color="rgba(255,255,255,0.3)", annotation_text="Even 50-50")

    fig_wp.update_layout(
        title=f"{selected_match_label} — Live Win Probability Timeline",
        xaxis_title="Match Delivery Number",
        yaxis_title="Batting Team Win Probability",
        yaxis=dict(range=[0, 1], tickformat=".0%"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#f8f9fa",
        height=420,
        margin=dict(l=10, r=10, t=40, b=20),
    )
    st.plotly_chart(fig_wp, width="stretch")

    # Game Turning Points Table (Top 5 largest absolute WPA)
    st.markdown("#### ⚡ Top 5 Match-Deciding Turning Points (Highest |WPA|)")
    top_wpa = match_df.sort_values(by="wpa", key=abs, ascending=False).head(5)
    display_wpa = top_wpa[
        [
            "innings",
            "overid",
            "over_ball_num",
            "striker_name",
            "bowler_name",
            "runs",
            "is_wicket",
            "wpa",
            "leverage_index",
        ]
    ].rename(
        columns={
            "innings": "Innings",
            "overid": "Over",
            "over_ball_num": "Ball",
            "striker_name": "Striker",
            "bowler_name": "Bowler",
            "runs": "Runs",
            "is_wicket": "Wicket?",
            "wpa": "Δ WPA",
            "leverage_index": "Leverage Index",
        }
    )
    st.dataframe(
        display_wpa.style.format({"Δ WPA": "{:+.4f}", "Leverage Index": "{:.2f}"}).background_gradient(
            subset=["Δ WPA"], cmap="coolwarm"
        ),
        width="stretch",
        hide_index=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 6: METHODOLOGY & VALIDATION ENGINE
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📖 Methodology & Validation Engine":
    st.markdown("### 📖 Machine Learning Architecture & Technical Audit")
    st.markdown("Transparent documentation of the statistical and predictive foundation powering PressureIQ.")

    # Model Evaluation Metrics
    df_eval = load_model_metrics()

    st.markdown("#### 1. Win Probability Model Performance")
    col_e1, col_e2, col_e3, col_e4 = st.columns(4)
    with col_e1:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-title">Validation ROC-AUC</div>
                <div class="metric-val" style="color: #10b981;">0.8161</div>
                <div style="font-size: 0.85rem; color: #94a3b8;">Ensemble (LR + XGB)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_e2:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-title">Validation Brier Score</div>
                <div class="metric-val" style="color: #00d4ff;">0.1734</div>
                <div style="font-size: 0.85rem; color: #94a3b8;">Strict Calibration Metric</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_e3:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-title">Holdout Test ROC-AUC</div>
                <div class="metric-val" style="color: #ffd700;">0.7775</div>
                <div style="font-size: 0.85rem; color: #94a3b8;">Chronological 2026 Test</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_e4:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-title">Test Brier Score</div>
                <div class="metric-val" style="color: #a78bfa;">0.1943</div>
                <div style="font-size: 0.85rem; color: #94a3b8;">No Leakage Observed</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("#### 2. Model Benchmark Comparison")
    st.dataframe(
        df_eval.style.format(
            {
                "brier_score": "{:.4f}",
                "log_loss": "{:.4f}",
                "roc_auc": "{:.4f}",
                "accuracy": "{:.2%}",
                "precision": "{:.2%}",
                "recall": "{:.2%}",
            }
        ),
        width="stretch",
        hide_index=True,
    )

    # Calibration Curve Image
    cal_img_path = FIGURES_DIR / "calibration_curves.png"
    if cal_img_path.exists():
        st.markdown("#### 3. Probability Calibration Curves (Innings 1 & 2)")
        st.image(str(cal_img_path), caption="Reliability Diagram: Predicted Probability vs Empirical Win Frequency")

    # Audit Findings Checklist
    st.markdown("#### 4. Pre-Launch Technical Audit Summary")
    st.markdown(
        """
        - ✅ **Zero Data Leakage:** Confirmed all prediction and pressure flags utilize strictly pre-delivery state variables.
        - ✅ **Volume Bias Verification:** Bowler rankings exhibit virtually **zero volume correlation** (Spearman $r = -0.025$, $p=0.69$). Batting rankings maintain controlled volume weighting ($r = 0.32$), correctly rewarding total match-winning value over time (analogous to Baseball WAR).
        - ✅ **Score Directionality:** Economy suppression ($-\\Delta \\text{Econ}$) directly boosts bowler score, rewarding elite dot-ball and boundary restrictions under pressure (e.g., Narine $-0.13$).
        - ✅ **Empirical Bayes Shrinkage:** All players with sample sizes $n < 50$ strictly excluded; players with moderate samples are smoothly regularized toward the prior population mean ($K=75.0$).
        - ✅ **28/28 Automated Tests Passing:** Validated continuously via pytest.
        """
    )

