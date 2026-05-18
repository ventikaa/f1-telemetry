import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pickle
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from features import build_features
from model import generate_insights
from simulation import simulate_strategy, find_optimal_window

# ── Palette ───────────────────────────────────────────────────────────────────
BALLROOM   = "#DAD2C8"   # off-white — main background
YELLOW     = "#F6BB02"   # Decor Yellow — tertiary accent
BLUE       = "#2A4C9E"   # Blue Sail — primary accent
RED        = "#BD1B1F"   # Red Inferno — secondary accent
BARBERA    = "#8B011A"   # Barbera — dark red
DARK_TEXT  = "#1A1A1A"   # near-black text on light bg
MID_TEXT   = "#4A4A4A"   # secondary text
SUBTLE     = "#C4BDB3"   # muted elements
BORDER     = "#B8B0A6"   # borders
CARD_BG    = "#EDE8E2"   # slightly darker than bg for cards
SIDEBAR_BG = "#D0C8BE"   # sidebar surface

TIRE_COLORS = {
    "SOFT":         RED,
    "MEDIUM":       YELLOW,
    "HARD":         "#6B6560",
    "INTERMEDIATE": "#2A7A4C",
    "WET":          BLUE,
}

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="F1 Telemetry",
    page_icon="🏎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;700&display=swap');

html, body, [class*="css"], .stApp {{
    font-family: 'JetBrains Mono', monospace !important;
    background-color: {BALLROOM} !important;
    color: {DARK_TEXT} !important;
}}

/* Sidebar */
section[data-testid="stSidebar"] {{
    background-color: {SIDEBAR_BG} !important;
    border-right: 2px solid {BLUE} !important;
}}
section[data-testid="stSidebar"] * {{
    font-family: 'JetBrains Mono', monospace !important;
    color: {DARK_TEXT} !important;
}}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stRadio label,
section[data-testid="stSidebar"] .stRadio div {{
    font-size: 10px !important;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: {MID_TEXT} !important;
}}

/* Selectbox */
.stSelectbox > div > div {{
    background-color: {BALLROOM} !important;
    border: 1px solid {BORDER} !important;
    color: {DARK_TEXT} !important;
    font-family: 'JetBrains Mono', monospace !important;
}}

/* Metric cards */
[data-testid="metric-container"] {{
    background: {CARD_BG};
    border: 1px solid {BORDER};
    border-top: 3px solid {BLUE};
    border-radius: 3px;
    padding: 16px 20px;
}}
[data-testid="metric-container"] label {{
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 9px !important;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: {MID_TEXT} !important;
}}
[data-testid="metric-container"] [data-testid="stMetricValue"] {{
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 28px !important;
    font-weight: 700;
    color: {DARK_TEXT} !important;
}}
[data-testid="metric-container"] [data-testid="stMetricDelta"] {{
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important;
}}

/* Headers */
h1, h2, h3 {{
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 700;
    letter-spacing: 0.02em;
    color: {DARK_TEXT} !important;
}}

/* Radio */
.stRadio > div {{ gap: 2px; }}
.stRadio label {{
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}}
/* Active radio highlight */
.stRadio [data-baseweb="radio"] input:checked + div {{
    border-color: {RED} !important;
}}

/* Slider */
.stSlider .st-ae {{ background: {BLUE} !important; }}
.stSlider * {{ font-family: 'JetBrains Mono', monospace !important; }}

/* Expander */
.streamlit-expanderHeader {{
    background: {CARD_BG} !important;
    border: 1px solid {BORDER} !important;
    color: {DARK_TEXT} !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}}

/* Divider */
hr {{ border-color: {BORDER} !important; }}

/* Main block */
.block-container {{ padding-top: 2rem; background-color: {BALLROOM} !important; }}

/* All text elements */
p, span, div, label {{
    font-family: 'JetBrains Mono', monospace !important;
}}
</style>
""", unsafe_allow_html=True)

# ── Plotly theme ──────────────────────────────────────────────────────────────
PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor=CARD_BG,
    font=dict(family="JetBrains Mono, monospace", color=DARK_TEXT, size=11),
    xaxis=dict(gridcolor=BORDER, linecolor=BORDER, tickcolor=BORDER, tickfont=dict(color=MID_TEXT)),
    yaxis=dict(gridcolor=BORDER, linecolor=BORDER, tickcolor=BORDER, tickfont=dict(color=MID_TEXT)),
    margin=dict(t=30, b=10, l=10, r=10),
)

# ── Data loading ──────────────────────────────────────────────────────────────
FEATURES_CSV = Path(__file__).parent.parent / "data" / "processed" / "features.csv"
MODEL_PKL    = Path(__file__).parent.parent / "data" / "processed" / "model.pkl"

@st.cache_data
def load_data():
    if not FEATURES_CSV.exists():
        st.error("Run src/pipeline.py then src/features.py first.")
        st.stop()
    return pd.read_csv(FEATURES_CSV)

@st.cache_resource
def load_model():
    if not MODEL_PKL.exists():
        st.error("Run src/model.py first.")
        st.stop()
    with open(MODEL_PKL, "rb") as f:
        return pickle.load(f)

df    = load_data()
model = load_model()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style='padding: 8px 0 24px 0;'>
        <div style='font-family: JetBrains Mono, monospace; font-size: 20px;
                    font-weight: 700; letter-spacing: 0.06em; color: {RED};'>
            F1_TELEMETRY
        </div>
        <div style='font-size: 9px; letter-spacing: 0.2em; color: {MID_TEXT};
                    text-transform: uppercase; margin-top: 4px;'>
            analytics &amp; strategy
        </div>
    </div>
    """, unsafe_allow_html=True)

    season  = st.selectbox("Season",  sorted(df["season"].unique(), reverse=True))
    races   = sorted(df[df["season"] == season]["race"].unique())
    race    = st.selectbox("Race",    races)
    drivers = sorted(df[(df["season"] == season) & (df["race"] == race)]["driver"].unique())
    driver  = st.selectbox("Driver",  drivers)

    st.markdown(f"<hr style='border-color:{BORDER};margin:16px 0;'>", unsafe_allow_html=True)

    section = st.radio("", ["Overview", "Telemetry", "Insights", "Strategy Sim"],
                       label_visibility="collapsed")

    st.markdown(f"""
    <div style='margin-top: 32px; font-size: 9px; letter-spacing: 0.1em;
                color: {SUBTLE}; text-transform: uppercase;'>
        {df['season'].nunique()} seasons · {df['race'].nunique()} races<br>
        {df['driver'].nunique()} drivers · {len(df):,} laps
    </div>
    """, unsafe_allow_html=True)

filtered  = df[(df["season"] == season) & (df["race"] == race)]
driver_df = filtered[filtered["driver"] == driver]

# ── Section header ────────────────────────────────────────────────────────────
def section_header(title, subtitle=""):
    st.markdown(f"""
    <div style='margin-bottom: 24px; border-bottom: 2px solid {BLUE}; padding-bottom: 12px;'>
        <span style='font-family: JetBrains Mono, monospace; font-size: 9px;
                     letter-spacing: 0.2em; color: {RED}; text-transform: uppercase;'>
            // {subtitle or f"season {season}"}
        </span>
        <div style='margin-top: 4px; font-family: JetBrains Mono, monospace;
                    font-size: 28px; font-weight: 700; color: {DARK_TEXT};
                    letter-spacing: 0.02em;'>
            {title}
        </div>
    </div>
    """, unsafe_allow_html=True)

def subheader(text):
    st.markdown(f"""
    <div style='font-family: JetBrains Mono, monospace; font-size: 10px;
                letter-spacing: 0.15em; color: {BLUE}; text-transform: uppercase;
                margin: 20px 0 8px 0; border-left: 3px solid {RED}; padding-left: 10px;'>
        {text}
    </div>
    """, unsafe_allow_html=True)

# ── OVERVIEW ──────────────────────────────────────────────────────────────────
if section == "Overview":
    section_header(f"{race} Grand Prix", f"season {season}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg lap time",  f"{driver_df['lap_time_s'].mean():.2f}s")
    c2.metric("Deg rate",      f"{driver_df['deg_rate'].mean():+.3f}s/lap")
    c3.metric("Consistency",   f"±{driver_df['consistency'].mean():.2f}s")
    best_pos = driver_df["position"].min()
    c4.metric("Best position", str(int(best_pos)) if not np.isnan(best_pos) else "–")

    st.markdown("<br>", unsafe_allow_html=True)
    col_l, col_r = st.columns([3, 2])

    with col_l:
        subheader("Driver comparison — avg lap time")
        avg_times = filtered.groupby("driver")["lap_time_s"].mean().sort_values().reset_index()
        colors = [RED if d == driver else BLUE for d in avg_times["driver"]]
        fig = go.Figure(go.Bar(
            x=avg_times["driver"], y=avg_times["lap_time_s"],
            marker_color=colors, marker_line_width=0,
        ))
        fig.update_layout(**PLOT_LAYOUT, showlegend=False,
                          yaxis_title="avg lap time (s)", xaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        subheader(f"Tire usage — {driver}")
        tire_counts = driver_df.groupby("tire")["lap"].count().reset_index()
        tire_counts.columns = ["tire", "laps"]
        fig2 = go.Figure(go.Pie(
            labels=tire_counts["tire"],
            values=tire_counts["laps"],
            marker_colors=[TIRE_COLORS.get(t, "#888") for t in tire_counts["tire"]],
            hole=0.55,
            textfont=dict(family="JetBrains Mono", size=11, color=DARK_TEXT),
        ))
        fig2.update_layout(**PLOT_LAYOUT, showlegend=True,
                           legend=dict(font=dict(color=DARK_TEXT, family="JetBrains Mono")))
        st.plotly_chart(fig2, use_container_width=True)

# ── TELEMETRY ─────────────────────────────────────────────────────────────────
elif section == "Telemetry":
    section_header(f"{driver} — {race}", f"season {season} telemetry")

    other_drivers = [d for d in drivers if d != driver]
    compare_driver = st.selectbox("Compare with", ["None"] + other_drivers)

    subheader("Lap time progression")
    fig = go.Figure()
    for compound in driver_df["tire"].unique():
        sub = driver_df[driver_df["tire"] == compound]
        fig.add_trace(go.Scatter(
            x=sub["lap"], y=sub["lap_time_s"],
            mode="markers+lines", name=f"{driver} / {compound}",
            marker=dict(color=TIRE_COLORS.get(compound, "#888"), size=5),
            line=dict(color=TIRE_COLORS.get(compound, "#888"), width=2),
        ))
    if compare_driver != "None":
        cmp_df = filtered[filtered["driver"] == compare_driver]
        for compound in cmp_df["tire"].unique():
            sub = cmp_df[cmp_df["tire"] == compound]
            fig.add_trace(go.Scatter(
                x=sub["lap"], y=sub["lap_time_s"],
                mode="markers+lines", name=f"{compare_driver} / {compound}",
                marker=dict(color=TIRE_COLORS.get(compound, "#888"), size=5, symbol="diamond"),
                line=dict(color=TIRE_COLORS.get(compound, "#888"), width=2, dash="dot"),
            ))
    fig.update_layout(**PLOT_LAYOUT, xaxis_title="lap", yaxis_title="lap time (s)",
                      legend=dict(font=dict(color=DARK_TEXT, family="JetBrains Mono")))
    st.plotly_chart(fig, use_container_width=True)

    col_l, col_r = st.columns(2)
    with col_l:
        subheader("Tire degradation curves")
        fig3 = go.Figure()
        for compound in filtered["tire"].unique():
            sub = (filtered[filtered["tire"] == compound]
                   .groupby("tire_age")["lap_time_s"].mean().reset_index())
            fig3.add_trace(go.Scatter(
                x=sub["tire_age"], y=sub["lap_time_s"],
                mode="lines", name=compound,
                line=dict(color=TIRE_COLORS.get(compound, "#888"), width=2.5),
            ))
        fig3.update_layout(**PLOT_LAYOUT,
                           xaxis_title="tire age (laps)", yaxis_title="avg lap time (s)",
                           legend=dict(font=dict(color=DARK_TEXT, family="JetBrains Mono")))
        st.plotly_chart(fig3, use_container_width=True)

    with col_r:
        subheader("Gap to leader")
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(
            x=driver_df["lap"], y=driver_df["gap_to_leader"],
            mode="lines", fill="tozeroy",
            line=dict(color=BLUE, width=2),
            fillcolor=f"rgba(42,76,158,0.15)",
            name="gap to leader",
        ))
        fig4.update_layout(**PLOT_LAYOUT,
                           xaxis_title="lap", yaxis_title="gap (s)",
                           legend=dict(font=dict(color=DARK_TEXT, family="JetBrains Mono")))
        st.plotly_chart(fig4, use_container_width=True)

# ── INSIGHTS ──────────────────────────────────────────────────────────────────
elif section == "Insights":
    section_header(f"{race} Grand Prix", f"season {season} insights")

    insights = generate_insights(filtered)
    for i, insight in enumerate(insights):
        st.markdown(f"""
        <div style='background:{CARD_BG}; border-left: 3px solid {BLUE};
                    padding: 14px 18px; margin-bottom: 10px; border-radius: 2px;
                    border: 1px solid {BORDER};'>
            <span style='font-size: 9px; letter-spacing: 0.15em; color: {RED};
                         text-transform: uppercase; font-family: JetBrains Mono, monospace;'>
                // insight_{i+1:02d}
            </span>
            <div style='margin-top: 6px; font-size: 13px; color: {DARK_TEXT};
                        line-height: 1.6; font-family: JetBrains Mono, monospace;'>
                {insight}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_l, col_r = st.columns(2)

    with col_l:
        subheader("Consistency leaderboard")
        cons_df = (filtered.groupby("driver")["consistency"].mean()
                   .sort_values().reset_index())
        colors = [RED if d == driver else BLUE for d in cons_df["driver"]]
        fig = go.Figure(go.Bar(
            x=cons_df["driver"], y=cons_df["consistency"],
            marker_color=colors, marker_line_width=0,
        ))
        fig.update_layout(**PLOT_LAYOUT, showlegend=False,
                          yaxis_title="std dev (s) ↓ better")
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        subheader("Degradation by compound")
        deg_df = (filtered.groupby("tire")["deg_rate"].mean()
                  .reset_index().sort_values("deg_rate", ascending=False))
        fig2 = go.Figure(go.Bar(
            x=deg_df["tire"], y=deg_df["deg_rate"],
            marker_color=[TIRE_COLORS.get(t, "#888") for t in deg_df["tire"]],
            marker_line_width=0,
        ))
        fig2.update_layout(**PLOT_LAYOUT, showlegend=False,
                           yaxis_title="avg deg rate (s/lap)")
        st.plotly_chart(fig2, use_container_width=True)

# ── STRATEGY SIM ──────────────────────────────────────────────────────────────
elif section == "Strategy Sim":
    section_header("Strategy Simulator", f"season {season} · {race}")

    col_l, col_r = st.columns([1, 1])
    with col_l:
        st.markdown(f"""<div style='font-size:9px; letter-spacing:0.15em; color:{BLUE};
                        text-transform:uppercase; margin-bottom:12px;
                        font-family: JetBrains Mono, monospace;'>
                        // race parameters</div>""", unsafe_allow_html=True)
        race_laps  = st.slider("Race length (laps)", 40, 78, 57)
        compound_1 = st.selectbox("Stint 1 compound", ["SOFT", "MEDIUM", "HARD"])
        compound_2 = st.selectbox("Stint 2 compound", ["HARD", "MEDIUM", "SOFT"])
        pit_lap    = st.slider("Pit lap", 5, race_laps - 5, 17)
        noise      = st.slider("Randomness σ (sec)", 0.0, 1.5, 0.0, 0.1)

    result     = simulate_strategy(model, race_laps, pit_lap,
                                   compound_1=compound_1, compound_2=compound_2, noise=noise)
    mins, secs = divmod(result["total_s"], 60)

    with col_r:
        st.markdown(f"""
        <div style='background:{CARD_BG}; border: 1px solid {BORDER};
                    border-top: 3px solid {RED}; border-left: 3px solid {BLUE};
                    padding: 24px; border-radius: 3px; margin-top: 28px;'>
            <div style='font-size:9px; letter-spacing:0.15em; color:{MID_TEXT};
                        text-transform:uppercase; font-family: JetBrains Mono, monospace;'>
                // predicted race time
            </div>
            <div style='font-family: JetBrains Mono, monospace; font-size: 48px;
                        font-weight: 700; color: {DARK_TEXT}; line-height: 1.1; margin-top: 8px;'>
                {int(mins)}:{secs:05.2f}
            </div>
            <div style='margin-top: 12px; font-size: 11px; color: {MID_TEXT};
                        font-family: JetBrains Mono, monospace;'>
                {compound_1} → pit_lap={pit_lap} → {compound_2}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    subheader("Pit window sweep")

    window_df  = find_optimal_window(model, race_laps,
                                     compound_1=compound_1, compound_2=compound_2)
    best_pit   = int(window_df.iloc[0]["pit_lap"])
    your_delta = window_df[window_df["pit_lap"] == pit_lap]["delta_s"].values

    if pit_lap == best_pit:
        st.markdown(f"""<div style='background:#E8F5E8; border-left: 3px solid #2A7A4C;
                        padding: 12px 16px; border-radius: 2px; margin-bottom: 12px;
                        font-family: JetBrains Mono, monospace; font-size: 12px; color: {DARK_TEXT};'>
                        ✓ optimal pit lap: <strong>lap {best_pit}</strong> — perfect call.
                        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""<div style='background:#F5E8E8; border-left: 3px solid {RED};
                        padding: 12px 16px; border-radius: 2px; margin-bottom: 12px;
                        font-family: JetBrains Mono, monospace; font-size: 12px; color: {DARK_TEXT};'>
                        ✗ optimal pit lap: <strong>lap {best_pit}</strong>
                        — your pick costs <strong style='color:{RED};'>+{your_delta[0]:.1f}s</strong>
                        </div>""", unsafe_allow_html=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=window_df["pit_lap"], y=window_df["delta_s"],
        mode="lines", fill="tozeroy",
        line=dict(color=BLUE, width=2),
        fillcolor="rgba(42,76,158,0.1)",
        name="time delta",
    ))
    fig.add_vline(x=best_pit, line_dash="dash", line_color=BLUE,
                  annotation_text=f"optimal: lap {best_pit}",
                  annotation_font_color=BLUE,
                  annotation_font_family="JetBrains Mono")
    fig.add_vline(x=pit_lap, line_dash="dot", line_color=RED,
                  annotation_text=f"your pick: lap {pit_lap}",
                  annotation_font_color=RED,
                  annotation_font_family="JetBrains Mono")
    fig.update_layout(**PLOT_LAYOUT,
                      xaxis_title="pit lap", yaxis_title="time delta vs optimal (s)",
                      legend=dict(font=dict(color=DARK_TEXT, family="JetBrains Mono")))
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("// per-lap breakdown"):
        fig2 = px.line(result["lap_times"], x="lap", y="lap_time_s", color="tire",
                       color_discrete_map=TIRE_COLORS,
                       labels={"lap_time_s": "predicted lap time (s)"})
        fig2.add_vline(x=pit_lap, line_dash="dash", line_color=BORDER,
                       annotation_text=f"pit lap {pit_lap}",
                       annotation_font_family="JetBrains Mono")
        fig2.update_layout(**PLOT_LAYOUT,
                           legend=dict(font=dict(color=DARK_TEXT, family="JetBrains Mono")))
        st.plotly_chart(fig2, use_container_width=True)