"""
Africa Development Trajectory Forecaster — Streamlit Dashboard
"Where will Kenya be in 2035?"
"""

import io
import os
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# ── Constants ────────────────────────────────────────────────────────────────

COUNTRIES = {
    "KE": "Kenya",
    "UG": "Uganda",
    "TZ": "Tanzania",
    "ET": "Ethiopia",
    "RW": "Rwanda",
    "NG": "Nigeria",
    "GH": "Ghana",
    "SN": "Senegal",
    "ZA": "South Africa",
    "EG": "Egypt",
    "MA": "Morocco",
    "TN": "Tunisia",
    "MZ": "Mozambique",
    "ZM": "Zambia",
    "ZW": "Zimbabwe",
}

INDICATORS = {
    "NY.GDP.PCAP.KD": "GDP per Capita (const. 2015 USD)",
    "EG.ELC.ACCS.ZS": "Electricity Access (%)",
    "IT.NET.USER.ZS": "Internet Users (%)",
    "SP.DYN.LE00.IN": "Life Expectancy (years)",
    "SE.ADT.LITR.ZS": "Adult Literacy Rate (%)",
    "SH.DYN.MORT": "Under-5 Mortality (per 1,000)",
    "SP.URB.TOTL.IN.ZS": "Urbanisation (%)",
    "NE.TRD.GNFS.ZS": "Trade Openness (% GDP)",
}

SPARSE_INDICATORS = {"SE.ADT.LITR.ZS"}
SPARSE_THRESHOLD = 4

COUNTRY_COLORS = {
    "KE": "#00d26a",
    "UG": "#f5a623",
    "TZ": "#4299e1",
    "ET": "#9f7aea",
    "RW": "#ed64a6",
    "NG": "#f56565",
    "GH": "#48bb78",
    "SN": "#ed8936",
    "ZA": "#667eea",
    "EG": "#fc8181",
    "MA": "#68d391",
    "TN": "#63b3ed",
    "MZ": "#b794f4",
    "ZM": "#fbd38d",
    "ZW": "#a0aec0",
}

CLUSTER_COLORS = {
    "Advanced": "#00d26a",
    "Emerging": "#f5a623",
    "Developing": "#4299e1",
}

BG = "#060b17"
SURFACE = "#0d1526"
ELEVATED = "#152035"
ACCENT = "#00d26a"
GOLD = "#f5a623"
TEXT = "#e2e8f0"
MUTED = "#718096"

PROCESSED_DIR = os.path.join("data", "processed")


# ── Helpers ──────────────────────────────────────────────────────────────────

def hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def plotly_layout(title: str = "", height: int = 500) -> dict:
    return dict(
        title=dict(text=title, font=dict(color=TEXT, size=16, family="DM Sans")),
        height=height,
        paper_bgcolor=SURFACE,
        plot_bgcolor=ELEVATED,
        font=dict(color=TEXT, family="DM Sans"),
        xaxis=dict(
            gridcolor=hex_to_rgba("#ffffff", 0.05),
            zerolinecolor=hex_to_rgba("#ffffff", 0.1),
            tickfont=dict(color=MUTED),
        ),
        yaxis=dict(
            gridcolor=hex_to_rgba("#ffffff", 0.05),
            zerolinecolor=hex_to_rgba("#ffffff", 0.1),
            tickfont=dict(color=MUTED),
        ),
        legend=dict(
            bgcolor=hex_to_rgba(BG, 0.8),
            bordercolor=hex_to_rgba("#ffffff", 0.1),
            borderwidth=1,
            font=dict(color=TEXT, size=11),
        ),
        margin=dict(l=60, r=20, t=50, b=50),
        hovermode="x unified",
    )


# ── Data Loaders ─────────────────────────────────────────────────────────────

@st.cache_data
def load_development() -> pd.DataFrame:
    path = os.path.join(PROCESSED_DIR, "development.parquet")
    if not os.path.exists(path):
        st.error("development.parquet not found. Run `python data_pipeline.py` first.")
        st.stop()
    return pd.read_parquet(path)


@st.cache_data
def load_clusters() -> pd.DataFrame:
    path = os.path.join(PROCESSED_DIR, "clusters.parquet")
    if not os.path.exists(path):
        st.error("clusters.parquet not found. Run `python data_pipeline.py` first.")
        st.stop()
    return pd.read_parquet(path)


@st.cache_data
def load_kenya_gdp_forecast() -> pd.DataFrame | None:
    path = os.path.join(PROCESSED_DIR, "forecast_NY.GDP.PCAP.KD_KE.parquet")
    if not os.path.exists(path):
        return None
    return pd.read_parquet(path)


@st.cache_data
def load_forecast(indicator_code: str, iso2: str) -> pd.DataFrame | None:
    safe_ind = indicator_code.replace(".", "_")
    path = os.path.join(PROCESSED_DIR, f"forecast_{safe_ind}_{iso2}.parquet")
    if not os.path.exists(path):
        return None
    return pd.read_parquet(path)


@st.cache_data
def load_all_forecasts(indicator_code: str, iso2_list: list) -> dict:
    result = {}
    for iso2 in iso2_list:
        df = load_forecast(indicator_code, iso2)
        if df is not None:
            result[iso2] = df
    return result


def get_indicator_obs_count(dev_df: pd.DataFrame, iso2: str, indicator_code: str) -> int:
    if indicator_code not in dev_df.columns:
        return 0
    return int(dev_df[(dev_df["iso2"] == iso2) & dev_df[indicator_code].notna()].shape[0])


def is_sparse_for_country(dev_df: pd.DataFrame, iso2: str, indicator_code: str) -> bool:
    return get_indicator_obs_count(dev_df, iso2, indicator_code) < SPARSE_THRESHOLD


# ── CSS Injection ─────────────────────────────────────────────────────────────

def inject_css():
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

        /* Hide sidebar toggle and Material Icons */
        [data-testid="collapsedControl"],
        [data-testid="stSidebarCollapseButton"],
        [data-testid="stSidebarCollapsedControl"],
        button[aria-label="Close sidebar"],
        button[aria-label="Open sidebar"] {{
            display: none !important;
        }}
        span.material-symbols-rounded,
        span.material-symbols-outlined,
        span.material-icons {{
            visibility: hidden !important;
            font-size: 0 !important;
        }}

        /* Root theme */
        html, body, [data-testid="stApp"] {{
            background-color: {BG};
            color: {TEXT};
            font-family: 'DM Sans', sans-serif;
        }}
        [data-testid="stSidebar"] {{
            background-color: {SURFACE} !important;
            border-right: 1px solid {hex_to_rgba('#ffffff', 0.07)} !important;
        }}
        [data-testid="stSidebar"] * {{
            color: {TEXT} !important;
        }}

        /* Header */
        .dashboard-header {{
            padding: 2rem 0 1.5rem 0;
            border-bottom: 1px solid {hex_to_rgba(ACCENT, 0.25)};
            margin-bottom: 2rem;
        }}
        .dashboard-title {{
            font-family: 'Space Grotesk', sans-serif;
            font-size: 2rem;
            font-weight: 700;
            color: {TEXT};
            letter-spacing: -0.02em;
            line-height: 1.2;
        }}
        .dashboard-title span {{
            color: {ACCENT};
        }}
        .dashboard-subtitle {{
            color: {MUTED};
            font-size: 0.95rem;
            margin-top: 0.4rem;
        }}

        /* KPI cards */
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 1rem;
            margin: 1.5rem 0;
        }}
        .kpi-card {{
            background: {SURFACE};
            border: 1px solid {hex_to_rgba('#ffffff', 0.07)};
            border-radius: 12px;
            padding: 1.2rem 1.5rem;
            position: relative;
            overflow: hidden;
        }}
        .kpi-card::before {{
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 3px;
            background: linear-gradient(90deg, {ACCENT}, transparent);
        }}
        .kpi-label {{
            font-size: 0.75rem;
            font-weight: 500;
            color: {MUTED};
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }}
        .kpi-value {{
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.7rem;
            font-weight: 700;
            color: {ACCENT};
            margin: 0.3rem 0;
            line-height: 1;
        }}
        .kpi-delta {{
            font-size: 0.8rem;
            color: {MUTED};
        }}
        .kpi-delta.positive {{ color: {ACCENT}; }}
        .kpi-delta.negative {{ color: #fc8181; }}

        /* Executive summary */
        .exec-summary {{
            background: linear-gradient(135deg,
                {hex_to_rgba(ACCENT, 0.05)},
                {hex_to_rgba(GOLD, 0.03)}
            );
            border: 1px solid {hex_to_rgba(ACCENT, 0.2)};
            border-left: 4px solid {ACCENT};
            border-radius: 8px;
            padding: 1rem 1.5rem;
            font-size: 0.92rem;
            line-height: 1.7;
            color: {TEXT};
            margin: 1rem 0;
        }}
        .exec-summary strong {{
            color: {ACCENT};
        }}

        /* Cluster badge */
        .cluster-badge {{
            display: inline-block;
            padding: 0.2rem 0.65rem;
            border-radius: 999px;
            font-size: 0.72rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }}
        .cluster-Advanced {{ background: {hex_to_rgba(ACCENT, 0.15)}; color: {ACCENT}; }}
        .cluster-Emerging {{ background: {hex_to_rgba(GOLD, 0.15)}; color: {GOLD}; }}
        .cluster-Developing {{ background: {hex_to_rgba('#4299e1', 0.15)}; color: #4299e1; }}

        /* Tabs */
        [data-testid="stTabs"] [data-testid="stTab"] {{
            font-family: 'DM Sans', sans-serif !important;
            font-weight: 500 !important;
            color: {MUTED} !important;
            border-bottom: 2px solid transparent !important;
            padding: 0.6rem 1rem !important;
        }}
        [data-testid="stTabs"] [data-testid="stTab"][aria-selected="true"] {{
            color: {ACCENT} !important;
            border-bottom-color: {ACCENT} !important;
        }}
        [data-testid="stTabsContent"] {{
            background: transparent !important;
        }}

        /* Selectbox, multiselect */
        [data-testid="stMultiSelect"] > div > div,
        [data-testid="stSelectbox"] > div > div {{
            background: {ELEVATED} !important;
            border-color: {hex_to_rgba('#ffffff', 0.1)} !important;
            color: {TEXT} !important;
        }}

        /* Slider */
        [data-testid="stSlider"] .rc-slider-track {{ background: {ACCENT} !important; }}
        [data-testid="stSlider"] .rc-slider-handle {{
            border-color: {ACCENT} !important;
            background: {ACCENT} !important;
        }}

        /* Divider */
        hr {{ border-color: {hex_to_rgba('#ffffff', 0.07)} !important; }}

        /* Insufficient data notice */
        .no-forecast-notice {{
            background: {hex_to_rgba(GOLD, 0.06)};
            border: 1px solid {hex_to_rgba(GOLD, 0.25)};
            border-radius: 8px;
            padding: 0.9rem 1.2rem;
            font-size: 0.88rem;
            color: {GOLD};
            margin: 0.5rem 0;
        }}

        /* Literacy static table */
        .literacy-callout {{
            background: {SURFACE};
            border: 1px solid {hex_to_rgba('#ffffff', 0.08)};
            border-radius: 10px;
            padding: 1.25rem;
            margin: 0.75rem 0;
        }}
        .literacy-callout h4 {{
            color: {GOLD};
            font-size: 0.85rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.07em;
            margin-bottom: 0.6rem;
        }}

        /* Plotly chart container */
        .js-plotly-plot {{ border-radius: 10px; overflow: hidden; }}

        /* Download button */
        [data-testid="stDownloadButton"] > button {{
            background: {hex_to_rgba(ACCENT, 0.12)} !important;
            border: 1px solid {hex_to_rgba(ACCENT, 0.3)} !important;
            color: {ACCENT} !important;
            font-weight: 600 !important;
            border-radius: 8px !important;
        }}
        [data-testid="stDownloadButton"] > button:hover {{
            background: {hex_to_rgba(ACCENT, 0.2)} !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ── Tab 1: 10-Year Trajectory ─────────────────────────────────────────────────

def render_trajectory_tab(
    dev_df: pd.DataFrame,
    cluster_df: pd.DataFrame,
    selected_countries: list,
    indicator_code: str,
    horizon: int,
):
    ind_label = INDICATORS[indicator_code]
    is_sparse = indicator_code in SPARSE_INDICATORS

    # ── Executive Summary banner (Tab 1 top) ───────────────────────────────
    ke_cluster_row = cluster_df[cluster_df["iso2"] == "KE"]
    ke_cluster_label = ke_cluster_row.iloc[0]["cluster_label"] if not ke_cluster_row.empty else "N/A"

    ke_gdp_current = None
    ke_gdp_year = None
    if "NY.GDP.PCAP.KD" in dev_df.columns:
        ke_hist_gdp = dev_df[(dev_df["iso2"] == "KE") & dev_df["NY.GDP.PCAP.KD"].notna()]
        if not ke_hist_gdp.empty:
            latest_ke = ke_hist_gdp.sort_values("year").iloc[-1]
            ke_gdp_current = latest_ke["NY.GDP.PCAP.KD"]
            ke_gdp_year = int(latest_ke["year"])

    ke_gdp_2033 = None
    ke_fc_gdp = load_kenya_gdp_forecast()
    if ke_fc_gdp is not None:
        ke_fc_gdp["_year"] = pd.to_datetime(ke_fc_gdp["ds"]).dt.year
        row_2033 = ke_fc_gdp[(ke_fc_gdp["is_forecast"] == True) & (ke_fc_gdp["_year"] == 2033)]
        if not row_2033.empty:
            ke_gdp_2033 = row_2033.iloc[0]["yhat"]

    za_gdp_current = None
    if "NY.GDP.PCAP.KD" in dev_df.columns:
        za_hist_gdp = dev_df[(dev_df["iso2"] == "ZA") & dev_df["NY.GDP.PCAP.KD"].notna()]
        if not za_hist_gdp.empty:
            za_gdp_current = za_hist_gdp.sort_values("year").iloc[-1]["NY.GDP.PCAP.KD"]

    if ke_gdp_current is not None and ke_gdp_2033 is not None:
        pct_banner = ((ke_gdp_2033 - ke_gdp_current) / abs(ke_gdp_current)) * 100
        za_bench = f"${za_gdp_current:,.0f}" if za_gdp_current is not None else "N/A"
        st.markdown(
            f"""<div style="background:#0d1a1a;border:1px solid #00d26a;border-radius:8px;padding:16px 20px;margin-bottom:20px;">
<p style="color:#00d26a;font-weight:700;margin:0 0 8px;">Africa Development Intelligence</p>
<p style="color:#e2e8f0;margin:0;line-height:1.7;">Kenya GDP per capita: <b>${ke_gdp_current:,.0f}</b> ({ke_gdp_year}) — forecast to reach <b>${ke_gdp_2033:,.0f}</b> by 2033 (<b>{pct_banner:+.1f}%</b>). Current cluster: <b>{ke_cluster_label}</b>. South Africa benchmark: <b>{za_bench}</b>.</p>
</div>""",
            unsafe_allow_html=True,
        )

    # ── Literacy special case ──────────────────────────────────────────────
    if is_sparse:
        st.info(
            "Adult Literacy Rate data is survey-based and published infrequently. "
            "Forecast unavailable for most countries."
        )
        st.markdown(
            f"""
            <div class="literacy-callout">
                <h4>Adult Literacy Rate — Survey-Based Indicator</h4>
                This indicator is survey-based and collected infrequently.
                Forecast models require ≥{SPARSE_THRESHOLD} observations per country.
                Latest available values are shown as a reference table below.
            </div>
            """,
            unsafe_allow_html=True,
        )
        if indicator_code in dev_df.columns:
            literacy_rows = []
            for iso2 in selected_countries:
                cdf = dev_df[(dev_df["iso2"] == iso2) & dev_df[indicator_code].notna()]
                obs = len(cdf)
                if obs < SPARSE_THRESHOLD and obs > 0:
                    st.warning(
                        f"{COUNTRIES.get(iso2, iso2)}: only {obs} observation(s) available — "
                        "forecast model cannot be generated."
                    )
                if not cdf.empty:
                    latest = cdf.sort_values("year").iloc[-1]
                    literacy_rows.append(
                        {
                            "Country": COUNTRIES.get(iso2, iso2),
                            "ISO2": iso2,
                            "Latest Year": int(latest["year"]),
                            "Adult Literacy Rate (%)": round(latest[indicator_code], 1),
                            "Observations": obs,
                        }
                    )
            if literacy_rows:
                st.dataframe(
                    pd.DataFrame(literacy_rows).set_index("Country"),
                    use_container_width=True,
                )
            else:
                st.info("No literacy data available for the selected countries.")
        return

    # ── Forecast chart ─────────────────────────────────────────────────────
    fig = go.Figure()
    cutoff_year = 2023
    end_year = cutoff_year + horizon

    forecast_cache = load_all_forecasts(indicator_code, selected_countries)

    any_forecast = False
    for iso2 in selected_countries:
        color = COUNTRY_COLORS.get(iso2, "#ffffff")
        economy = COUNTRIES.get(iso2, iso2)

        fc_df = forecast_cache.get(iso2)
        if fc_df is None:
            continue

        # Historical segment
        hist = fc_df[~fc_df["is_forecast"]].copy()
        hist["year"] = pd.to_datetime(hist["ds"]).dt.year
        hist = hist[hist["year"] <= cutoff_year]

        fig.add_trace(
            go.Scatter(
                x=hist["ds"],
                y=hist["yhat"],
                mode="lines",
                line=dict(color=color, width=2.5),
                name=economy,
                legendgroup=iso2,
                showlegend=True,
                hovertemplate=f"<b>{economy}</b><br>Year: %{{x|%Y}}<br>{ind_label}: %{{y:.1f}}<extra></extra>",
            )
        )

        # Forecast segment (trimmed to horizon)
        fcast = fc_df[fc_df["is_forecast"]].copy()
        fcast["year"] = pd.to_datetime(fcast["ds"]).dt.year
        fcast = fcast[fcast["year"] <= end_year]

        if fcast.empty:
            continue

        any_forecast = True

        # Confidence band
        fig.add_trace(
            go.Scatter(
                x=pd.concat([fcast["ds"], fcast["ds"].iloc[::-1]]),
                y=pd.concat([fcast["yhat_upper"], fcast["yhat_lower"].iloc[::-1]]),
                fill="toself",
                fillcolor=hex_to_rgba(color, 0.12),
                line=dict(color="rgba(0,0,0,0)"),
                hoverinfo="skip",
                showlegend=False,
                legendgroup=iso2,
            )
        )

        # Dashed forecast line
        fig.add_trace(
            go.Scatter(
                x=fcast["ds"],
                y=fcast["yhat"],
                mode="lines",
                line=dict(color=color, width=2.5, dash="dash"),
                showlegend=False,
                legendgroup=iso2,
                hovertemplate=f"<b>{economy} (forecast)</b><br>Year: %{{x|%Y}}<br>{ind_label}: %{{y:.1f}}<extra></extra>",
            )
        )

    # Forecast boundary line
    fig.add_vline(
        x=pd.Timestamp(f"{cutoff_year}-07-01"),
        line=dict(color=hex_to_rgba("#ffffff", 0.35), width=1.5, dash="dot"),
        annotation_text="Forecast →",
        annotation_font=dict(color=MUTED, size=10),
        annotation_position="top right",
    )

    # Milestone markers at 2030 and 2035 (if in range)
    for milestone_year, milestone_label in [(2030, "2030"), (2035, "2035")]:
        if milestone_year <= end_year:
            fig.add_vline(
                x=pd.Timestamp(f"{milestone_year}-01-01"),
                line=dict(color=hex_to_rgba(GOLD, 0.3), width=1, dash="longdash"),
                annotation_text=milestone_label,
                annotation_font=dict(color=GOLD, size=10),
                annotation_position="top left",
            )

    layout = plotly_layout(f"{ind_label} — Historical & {horizon}-Year Forecast", 520)
    layout.update(
        xaxis=dict(
            **layout.get("xaxis", {}),
            title=dict(text="Year", font=dict(color=MUTED)),
        ),
        yaxis=dict(
            **layout.get("yaxis", {}),
            title=dict(text=ind_label, font=dict(color=MUTED)),
        ),
    )
    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True)

    if not any_forecast:
        st.markdown(
            '<div class="no-forecast-notice">No forecast data available for the selected countries and indicator.</div>',
            unsafe_allow_html=True,
        )
        return

    # ── KPI Cards for Kenya ────────────────────────────────────────────────
    ke_fc = forecast_cache.get("KE")
    if ke_fc is not None and indicator_code in dev_df.columns:
        ke_hist = dev_df[(dev_df["iso2"] == "KE") & dev_df[indicator_code].notna()]
        current_val = None
        if not ke_hist.empty:
            current_val = ke_hist.sort_values("year").iloc[-1][indicator_code]
            current_year = int(ke_hist.sort_values("year").iloc[-1]["year"])

        fc_2030 = None
        fc_2033 = None
        ke_fc["year"] = pd.to_datetime(ke_fc["ds"]).dt.year
        row_2030 = ke_fc[ke_fc["year"] == 2030]
        row_2033 = ke_fc[ke_fc["year"] == min(2033, end_year)]
        if not row_2030.empty:
            fc_2030 = row_2030.iloc[0]["yhat"]
        if not row_2033.empty:
            fc_2033 = row_2033.iloc[0]["yhat"]

        def fmt(v, code):
            if v is None:
                return "N/A"
            if code == "NY.GDP.PCAP.KD":
                return f"${v:,.0f}"
            elif code in ("EG.ELC.ACCS.ZS", "IT.NET.USER.ZS", "SE.ADT.LITR.ZS", "SP.URB.TOTL.IN.ZS", "NE.TRD.GNFS.ZS"):
                return f"{v:.1f}%"
            elif code == "SP.DYN.LE00.IN":
                return f"{v:.1f} yrs"
            elif code == "SH.DYN.MORT":
                return f"{v:.1f}"
            return f"{v:.2f}"

        pct_chg = None
        if current_val and fc_2033 and current_val != 0:
            pct_chg = ((fc_2033 - current_val) / abs(current_val)) * 100

        delta_class = ""
        if pct_chg is not None:
            # For mortality, decline is good
            if indicator_code == "SH.DYN.MORT":
                delta_class = "positive" if pct_chg < 0 else "negative"
            else:
                delta_class = "positive" if pct_chg >= 0 else "negative"

        delta_html = (
            f'<div class="kpi-delta {delta_class}">'
            + (f"{'▲' if pct_chg >= 0 else '▼'} {abs(pct_chg):.1f}% projected change"
               if pct_chg is not None else "Change: N/A")
            + "</div>"
        )

        st.markdown(
            f"""
            <div class="kpi-grid">
                <div class="kpi-card">
                    <div class="kpi-label">Kenya — Current ({current_year if current_val else '?'})</div>
                    <div class="kpi-value">{fmt(current_val, indicator_code)}</div>
                    <div class="kpi-delta">{ind_label}</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Kenya — 2030 Forecast</div>
                    <div class="kpi-value">{fmt(fc_2030, indicator_code)}</div>
                    <div class="kpi-delta">Projected value</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Kenya — {min(2033, end_year)} Forecast</div>
                    <div class="kpi-value">{fmt(fc_2033, indicator_code)}</div>
                    {delta_html}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Executive Summary ──────────────────────────────────────────────
        # EAC countries
        eac = ["KE", "UG", "TZ", "RW"]
        eac_vals = []
        for iso2 in eac:
            fc = forecast_cache.get(iso2)
            if fc is not None:
                fc["year"] = pd.to_datetime(fc["ds"]).dt.year
                row = fc[fc["year"] == min(2033, end_year)]
                if not row.empty:
                    eac_vals.append(row.iloc[0]["yhat"])
        eac_avg = sum(eac_vals) / len(eac_vals) if eac_vals else None

        za_fc = forecast_cache.get("ZA")
        za_val = None
        if za_fc is not None:
            za_fc["year"] = pd.to_datetime(za_fc["ds"]).dt.year
            za_row = za_fc[za_fc["year"] == min(2033, end_year)]
            if not za_row.empty:
                za_val = za_row.iloc[0]["yhat"]

        summary_parts = []
        if fc_2033 is not None:
            summary_parts.append(
                f"<strong>Kenya</strong> is projected to reach "
                f"<strong>{fmt(fc_2033, indicator_code)}</strong> by "
                f"<strong>{min(2033, end_year)}</strong>"
            )
        if za_val is not None:
            summary_parts.append(
                f"compared to South Africa at <strong>{fmt(za_val, indicator_code)}</strong>"
            )
        if eac_avg is not None:
            summary_parts.append(
                f"and an EAC average of <strong>{fmt(eac_avg, indicator_code)}</strong>"
            )

        if summary_parts:
            st.markdown(
                f'<div class="exec-summary">'
                + ", ".join(summary_parts)
                + "."
                + "</div>",
                unsafe_allow_html=True,
            )


# ── Tab 2: Country Clusters ───────────────────────────────────────────────────

def render_clusters_tab(dev_df: pd.DataFrame, cluster_df: pd.DataFrame):
    # Prepare bubble data
    gdp_col = "NY.GDP.PCAP.KD"
    le_col = "SP.DYN.LE00.IN"
    inet_col = "IT.NET.USER.ZS"

    plot_rows = []
    for iso2, economy in COUNTRIES.items():
        cdf = dev_df[dev_df["iso2"] == iso2].sort_values("year", ascending=False)
        gdp_val = cdf[gdp_col].dropna().iloc[0] if not cdf[gdp_col].dropna().empty else None
        le_val = cdf[le_col].dropna().iloc[0] if not cdf[le_col].dropna().empty else None
        inet_val = cdf[inet_col].dropna().iloc[0] if not cdf[inet_col].dropna().empty else 5.0

        cl_row = cluster_df[cluster_df["iso2"] == iso2]
        cluster_label = cl_row.iloc[0]["cluster_label"] if not cl_row.empty else "Developing"

        if gdp_val is not None and le_val is not None:
            plot_rows.append(
                {
                    "iso2": iso2,
                    "economy": economy,
                    "GDP per Capita": gdp_val,
                    "Life Expectancy": le_val,
                    "Internet Users (%)": max(inet_val, 2.0),
                    "cluster_label": cluster_label,
                }
            )

    scatter_df = pd.DataFrame(plot_rows)

    fig = go.Figure()

    for cluster in ["Advanced", "Emerging", "Developing"]:
        sub = scatter_df[scatter_df["cluster_label"] == cluster]
        if sub.empty:
            continue
        color = CLUSTER_COLORS[cluster]

        for _, row in sub.iterrows():
            is_kenya = row["iso2"] == "KE"
            symbol = "star" if is_kenya else "circle"
            size_scale = 18 if is_kenya else 14

            bubble_size = min(max(row["Internet Users (%)"] ** 0.55, 6), 35)

            fig.add_trace(
                go.Scatter(
                    x=[row["GDP per Capita"]],
                    y=[row["Life Expectancy"]],
                    mode="markers+text",
                    marker=dict(
                        size=bubble_size + (6 if is_kenya else 0),
                        color=hex_to_rgba(color, 0.75),
                        line=dict(
                            width=2 if is_kenya else 1,
                            color=color,
                        ),
                        symbol=symbol,
                    ),
                    text=[row["economy"]],
                    textposition="top center",
                    textfont=dict(
                        size=10 if not is_kenya else 12,
                        color=color if is_kenya else TEXT,
                        family="DM Sans",
                    ),
                    name=cluster,
                    legendgroup=cluster,
                    showlegend=(row.name == sub.index[0]),
                    hovertemplate=(
                        f"<b>{row['economy']}</b><br>"
                        f"GDP per Capita: ${{x:,.0f}}<br>"
                        f"Life Expectancy: {{y:.1f}} yrs<br>"
                        f"Internet Users: {row['Internet Users (%)']:.1f}%<br>"
                        f"Cluster: {cluster}<extra></extra>"
                    ),
                )
            )

    layout = plotly_layout("Development Clusters — GDP vs Life Expectancy", 560)
    layout.update(
        xaxis=dict(
            **layout.get("xaxis", {}),
            title=dict(text="GDP per Capita (const. 2015 USD)", font=dict(color=MUTED)),
            type="log",
        ),
        yaxis=dict(
            **layout.get("yaxis", {}),
            title=dict(text="Life Expectancy (years)", font=dict(color=MUTED)),
        ),
        hovermode="closest",
    )
    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True)

    st.caption("Bubble size = Internet Users (%). X-axis is log scale. ★ = Kenya.")

    # ── Cluster summary metrics ────────────────────────────────────────────
    gdp_col = "NY.GDP.PCAP.KD"
    cluster_summary = []
    for cluster in ["Advanced", "Emerging", "Developing"]:
        members_iso = cluster_df[cluster_df["cluster_label"] == cluster]["iso2"].tolist()
        count = len(members_iso)
        avg_gdps = []
        for iso2 in members_iso:
            cdf = dev_df[dev_df["iso2"] == iso2].sort_values("year", ascending=False)
            if gdp_col in cdf.columns:
                vals = cdf[gdp_col].dropna()
                if not vals.empty:
                    avg_gdps.append(vals.iloc[0])
        avg_gdp = sum(avg_gdps) / len(avg_gdps) if avg_gdps else None
        cluster_summary.append({
            "Cluster": cluster,
            "Countries": count,
            "Avg GDP per Capita (latest)": f"${avg_gdp:,.0f}" if avg_gdp is not None else "N/A",
        })

    summary_df = pd.DataFrame(cluster_summary).set_index("Cluster")
    st.markdown(
        f'<div style="margin: 1rem 0 0.4rem; font-size:0.8rem; font-weight:600; '
        f'color:{MUTED}; text-transform:uppercase; letter-spacing:0.07em;">Cluster Summary</div>',
        unsafe_allow_html=True,
    )
    st.dataframe(summary_df, use_container_width=True)

    # Cluster member table
    st.markdown("### Cluster Membership")
    cols = st.columns(3)
    for i, cluster in enumerate(["Advanced", "Emerging", "Developing"]):
        members = cluster_df[cluster_df["cluster_label"] == cluster]["economy"].tolist()
        color = CLUSTER_COLORS[cluster]
        with cols[i]:
            badge = f'<span class="cluster-badge cluster-{cluster}">{cluster}</span>'
            st.markdown(badge, unsafe_allow_html=True)
            for m in sorted(members):
                st.markdown(
                    f'<div style="padding: 0.2rem 0; color: {TEXT}; font-size: 0.88rem;">• {m}</div>',
                    unsafe_allow_html=True,
                )


# ── Tab 3: Development Benchmarks ────────────────────────────────────────────

def render_benchmarks_tab(dev_df: pd.DataFrame, cluster_df: pd.DataFrame):
    # Build latest-value matrix for heatmap
    BENCHMARK_INDS = [
        "NY.GDP.PCAP.KD",
        "EG.ELC.ACCS.ZS",
        "IT.NET.USER.ZS",
        "SP.DYN.LE00.IN",
        "SP.URB.TOTL.IN.ZS",
        "SH.DYN.MORT",
        "NE.TRD.GNFS.ZS",
    ]
    BENCHMARK_LABELS = {k: INDICATORS[k] for k in BENCHMARK_INDS}

    matrix_rows = {}
    for iso2, economy in COUNTRIES.items():
        cdf = dev_df[dev_df["iso2"] == iso2].sort_values("year", ascending=False)
        row = {}
        for ind in BENCHMARK_INDS:
            if ind in cdf.columns:
                vals = cdf[ind].dropna()
                row[ind] = vals.iloc[0] if not vals.empty else np.nan
            else:
                row[ind] = np.nan
        matrix_rows[economy] = row

    matrix_df = pd.DataFrame(matrix_rows).T
    matrix_df.index.name = "Country"

    # Normalise 0-1 (1 = best); for mortality, invert
    norm_df = matrix_df.copy()
    for ind in BENCHMARK_INDS:
        col = matrix_df[ind]
        col_min, col_max = col.min(), col.max()
        if col_max > col_min:
            if ind == "SH.DYN.MORT":
                norm_df[ind] = 1 - (col - col_min) / (col_max - col_min)
            else:
                norm_df[ind] = (col - col_min) / (col_max - col_min)
        else:
            norm_df[ind] = 0.5

    short_labels = [BENCHMARK_LABELS[i].replace(" (const. 2015 USD)", "").replace(" (per 1,000)", "").replace(" (% GDP)", "") for i in BENCHMARK_INDS]
    hover_vals = matrix_df.round(1).astype(str)

    fig_heat = px.imshow(
        norm_df.values,
        x=short_labels,
        y=norm_df.index.tolist(),
        color_continuous_scale=[
            [0.0, "#0d1526"],
            [0.25, "#164070"],
            [0.5, "#1a6b9a"],
            [0.75, "#00a85a"],
            [1.0, "#00d26a"],
        ],
        aspect="auto",
        zmin=0,
        zmax=1,
    )

    # Add hover text with raw values
    fig_heat.update_traces(
        customdata=hover_vals.values,
        hovertemplate="<b>%{y}</b><br>%{x}<br>Value: %{customdata}<br>Score: %{z:.2f}<extra></extra>",
    )

    fig_heat.update_layout(
        title=dict(
            text="Development Benchmark Heatmap (normalised, 1 = best in dataset)",
            font=dict(color=TEXT, size=14, family="DM Sans"),
        ),
        height=520,
        paper_bgcolor=SURFACE,
        plot_bgcolor=ELEVATED,
        font=dict(color=TEXT, family="DM Sans"),
        coloraxis_colorbar=dict(
            title="Score",
            tickfont=dict(color=MUTED),
            titlefont=dict(color=MUTED),
        ),
        margin=dict(l=120, r=20, t=60, b=100),
        xaxis=dict(tickangle=-35, tickfont=dict(color=TEXT, size=10)),
        yaxis=dict(tickfont=dict(color=TEXT, size=10)),
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    # ── Radar chart: Kenya vs Cluster average vs ZA ────────────────────────
    RADAR_INDS = [
        "NY.GDP.PCAP.KD",
        "EG.ELC.ACCS.ZS",
        "IT.NET.USER.ZS",
        "SP.DYN.LE00.IN",
        "SP.URB.TOTL.IN.ZS",
    ]
    RADAR_LABELS = [
        INDICATORS[i].replace(" (const. 2015 USD)", "")
        .replace(" (% GDP)", "")
        for i in RADAR_INDS
    ]

    def get_norm_row(iso2):
        return [norm_df.loc[COUNTRIES[iso2], ind] if COUNTRIES[iso2] in norm_df.index else 0 for ind in RADAR_INDS]

    ke_cluster_row = cluster_df[cluster_df["iso2"] == "KE"]
    ke_cluster = ke_cluster_row.iloc[0]["cluster_label"] if not ke_cluster_row.empty else "Developing"
    cluster_members = cluster_df[cluster_df["cluster_label"] == ke_cluster]["iso2"].tolist()

    cluster_avg = []
    for ind in RADAR_INDS:
        vals = [norm_df.loc[COUNTRIES[m], ind] for m in cluster_members if COUNTRIES[m] in norm_df.index]
        cluster_avg.append(np.mean(vals) if vals else 0)

    ke_vals = get_norm_row("KE")
    za_vals = get_norm_row("ZA")

    fig_radar = go.Figure()
    angles = RADAR_LABELS + [RADAR_LABELS[0]]

    for name, vals, color in [
        ("Kenya", ke_vals, COUNTRY_COLORS["KE"]),
        (f"{ke_cluster} Cluster Avg", cluster_avg, GOLD),
        ("South Africa (Benchmark)", za_vals, COUNTRY_COLORS["ZA"]),
    ]:
        closed_vals = vals + [vals[0]]
        fig_radar.add_trace(
            go.Scatterpolar(
                r=closed_vals,
                theta=angles,
                fill="toself",
                fillcolor=hex_to_rgba(color, 0.1),
                line=dict(color=color, width=2),
                name=name,
                hovertemplate="<b>%{theta}</b><br>Score: %{r:.2f}<extra></extra>",
            )
        )

    fig_radar.update_layout(
        title=dict(
            text=f"Kenya vs {ke_cluster} Cluster Avg vs South Africa",
            font=dict(color=TEXT, size=14, family="DM Sans"),
        ),
        height=450,
        paper_bgcolor=SURFACE,
        polar=dict(
            bgcolor=ELEVATED,
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                gridcolor=hex_to_rgba("#ffffff", 0.08),
                tickfont=dict(color=MUTED, size=9),
                linecolor=hex_to_rgba("#ffffff", 0.1),
            ),
            angularaxis=dict(
                tickfont=dict(color=TEXT, size=10, family="DM Sans"),
                linecolor=hex_to_rgba("#ffffff", 0.1),
                gridcolor=hex_to_rgba("#ffffff", 0.08),
            ),
        ),
        legend=dict(
            bgcolor=hex_to_rgba(BG, 0.8),
            bordercolor=hex_to_rgba("#ffffff", 0.1),
            borderwidth=1,
            font=dict(color=TEXT, size=11),
        ),
        font=dict(color=TEXT, family="DM Sans"),
        margin=dict(l=40, r=40, t=60, b=40),
    )
    st.plotly_chart(fig_radar, use_container_width=True)


# ── Tab 4: Export ─────────────────────────────────────────────────────────────

def render_export_tab(dev_df: pd.DataFrame, cluster_df: pd.DataFrame):
    st.markdown("### Export Data")
    st.markdown(
        "Download the full dataset — historical indicators, Prophet forecasts, and cluster assignments — as an Excel workbook.",
        unsafe_allow_html=False,
    )

    @st.cache_data
    def build_excel() -> bytes:
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            # Sheet 1: Historical
            dev_df.to_excel(writer, sheet_name="Historical", index=False)

            # Sheet 2: Clusters
            cluster_df.to_excel(writer, sheet_name="Clusters", index=False)

            # Sheet 3+: Forecasts per indicator
            for ind_code, ind_label in INDICATORS.items():
                if ind_code in SPARSE_INDICATORS:
                    continue
                safe_ind = ind_code.replace(".", "_")
                frames = []
                for iso2, economy in COUNTRIES.items():
                    path = os.path.join(
                        PROCESSED_DIR, f"forecast_{safe_ind}_{iso2}.parquet"
                    )
                    if os.path.exists(path):
                        df = pd.read_parquet(path)
                        df.insert(0, "country", economy)
                        df.insert(0, "iso2", iso2)
                        frames.append(df)
                if frames:
                    combined = pd.concat(frames, ignore_index=True)
                    sheet_name = ind_label[:28].strip()
                    combined.to_excel(writer, sheet_name=sheet_name, index=False)

        return output.getvalue()

    excel_data = build_excel()

    col1, col2 = st.columns([1, 2])
    with col1:
        st.download_button(
            label="Download Excel Workbook",
            data=excel_data,
            file_name="africa_development_forecast.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    with col2:
        st.markdown(
            f"""
            <div style="background:{SURFACE}; border:1px solid {hex_to_rgba('#fff', 0.07)};
                        border-radius:10px; padding:1rem 1.25rem; font-size:0.88rem; color:{MUTED};">
                <strong style="color:{TEXT};">Workbook contents:</strong><br>
                • <em>Historical</em> — {len(dev_df):,} rows across 15 countries × 24 years<br>
                • <em>Clusters</em> — K-Means cluster assignments for all 15 countries<br>
                • One sheet per indicator (excl. Literacy) with full Prophet forecast output
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("### Raw Data Preview — Historical")
    display_cols = ["iso2", "economy", "year"] + [
        c for c in dev_df.columns if c not in ["iso2", "economy", "year"]
    ]
    st.dataframe(dev_df[display_cols].sort_values(["iso2", "year"]), use_container_width=True, height=300)

    st.markdown("### Cluster Assignments")
    st.dataframe(
        cluster_df.sort_values("cluster_label")[
            ["iso2", "economy", "cluster_label", "cluster_id"]
            + [c for c in cluster_df.columns if c not in ["iso2", "economy", "cluster_label", "cluster_id"]]
        ],
        use_container_width=True,
    )


# ── Main App ──────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="Africa Dev Forecast",
        page_icon="🌍",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_css()

    # Load data
    dev_df = load_development()
    cluster_df = load_clusters()

    # ── Header ─────────────────────────────────────────────────────────────
    st.markdown(
        """
        <div class="dashboard-header">
            <div class="dashboard-title">
                Africa Development Trajectory <span>Forecaster</span>
            </div>
            <div class="dashboard-subtitle">
                Where will Kenya be in 2035? Powered by World Bank data · Prophet forecasting · K-Means clustering
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Sidebar ─────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(
            f'<div style="font-family:Space Grotesk; font-size:0.85rem; font-weight:600; '
            f'color:{ACCENT}; text-transform:uppercase; letter-spacing:0.1em; '
            f'margin-bottom:1.2rem;">Controls</div>',
            unsafe_allow_html=True,
        )

        country_options = {v: k for k, v in COUNTRIES.items()}
        default_names = ["Kenya", "South Africa", "Ghana", "Ethiopia", "Nigeria"]
        selected_names = st.multiselect(
            "Countries",
            options=sorted(country_options.keys()),
            default=default_names,
        )
        selected_countries = [country_options[n] for n in selected_names]
        if not selected_countries:
            selected_countries = ["KE"]

        ind_options = list(INDICATORS.values())
        ind_index = st.selectbox(
            "Indicator",
            options=ind_options,
            index=0,
        )
        indicator_code = [k for k, v in INDICATORS.items() if v == ind_index][0]

        horizon = st.slider("Forecast Horizon (years)", min_value=1, max_value=10, value=10)

        st.markdown("---")
        st.markdown(
            f'<div style="font-size:0.75rem; color:{MUTED}; line-height:1.6;">'
            "Data: World Bank API v2<br>"
            "Model: Facebook Prophet<br>"
            "Clustering: K-Means (k=3)<br>"
            f"Countries: {len(COUNTRIES)} · Years: 2000–{2023 + horizon}"
            "</div>",
            unsafe_allow_html=True,
        )

        # Kenya cluster badge
        ke_row = cluster_df[cluster_df["iso2"] == "KE"]
        if not ke_row.empty:
            ke_cluster = ke_row.iloc[0]["cluster_label"]
            st.markdown(
                f'<div style="margin-top:1rem;">'
                f'<span style="font-size:0.75rem; color:{MUTED};">Kenya cluster: </span>'
                f'<span class="cluster-badge cluster-{ke_cluster}">{ke_cluster}</span>'
                f"</div>",
                unsafe_allow_html=True,
            )

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(
        ["🚀 10-Year Trajectory", "🔵 Country Clusters", "🌡️ Development Benchmarks", "💾 Export"]
    )

    with tab1:
        render_trajectory_tab(
            dev_df, cluster_df, selected_countries, indicator_code, horizon
        )

    with tab2:
        render_clusters_tab(dev_df, cluster_df)

    with tab3:
        render_benchmarks_tab(dev_df, cluster_df)

    with tab4:
        render_export_tab(dev_df, cluster_df)


if __name__ == "__main__":
    main()
