# 🌍 Africa Development Trajectory Forecaster

**A production-grade Streamlit intelligence dashboard that fetches 8 World Bank development indicators for 15 African countries (2000–2023), trains 109 Prophet time-series models to forecast each indicator to 2033, clusters countries into 3 development tiers using K-Means, and surfaces the results through an interactive 4-tab dark dashboard — answering the core planning question: "Where will Kenya be in 2033?"**

| Metric | Value |
|---|---|
| Historical rows | 360 (15 countries × 24 years) |
| Year range | 2000 – 2023 |
| Indicators tracked | 8 |
| Prophet models trained | 109 |
| Forecast horizon | 10 years (to 2033) |
| K-Means clusters | 3 (Advanced / Emerging / Developing) |
| Dashboard tabs | 4 |
| Lines of code | 1,232 (app.py) |
| Data cost | $0 (World Bank REST API) |
| Dashboard cost | $0 (Streamlit Community Cloud) |

---

## 🎯 Project Goal

Development economists, policy analysts, and researchers need a single view that combines historical trajectories, future projections, and cross-country positioning for Sub-Saharan and North African economies. This project answers three specific questions:

1. **Trajectory** — How will Kenya's GDP per capita, electricity access, internet penetration, life expectancy, urbanisation, trade openness, and under-5 mortality evolve to 2033 under a data-driven Prophet forecast?
2. **Positioning** — Which development cluster does each country belong to, and how does Kenya compare to its EAC-5 peers and to South Africa as an upper-middle-income benchmark?
3. **Benchmarking** — Across all 8 indicators simultaneously, where does each country stand relative to the rest of the dataset?

The dashboard is designed to work with real World Bank data fetched fresh from the REST API — no synthetic values, no estimates, no interpolated series.

---

## 🧬 System Architecture

```
World Bank REST API v2
        │
        ▼
data_pipeline.py
  ├── fetch_indicator()       — paginated REST fetch per indicator per country
  ├── fetch_all_indicators()  — loops all 8 indicators, combines to long DataFrame
  ├── build_wide_table()      — pivot long→wide, enforce 15×24 grid
  ├── train_prophet_models()  — Prophet per (country, indicator) pair ≥8 obs
  │     └── forecast_<IND>_<ISO2>.parquet  (109 files)
  └── cluster_countries()     — StandardScaler → KMeans(k=3) → labelled clusters
        └── clusters.parquet

data/processed/
  ├── development.parquet     — 360-row wide table (historical)
  ├── clusters.parquet        — 15-row cluster assignments
  └── forecast_*.parquet      — 109 per-country per-indicator forecasts

app.py (Streamlit)
  ├── Tab 1: 10-Year Trajectory   — Prophet chart + KPI cards + EAC summary
  ├── Tab 2: Country Clusters     — Bubble scatter + radar chart + membership table
  ├── Tab 3: Development Benchmarks — Normalised heatmap + Kenya radar
  └── Tab 4: Export               — Excel workbook (Historical + Clusters + Forecasts)
```

The pipeline and the dashboard are fully decoupled. Run `data_pipeline.py` once to populate `data/processed/`; the Streamlit app reads only from those parquet files and never calls the World Bank API at runtime.

---

## 🛠️ Technical Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11+ |
| Data source | World Bank REST API v2 |
| Data storage | Parquet (PyArrow) |
| Forecasting | Facebook Prophet 1.1.5 |
| Clustering | scikit-learn KMeans + StandardScaler |
| Dashboard | Streamlit 1.45+ |
| Visualisation | Plotly 5.24+ |
| Export | openpyxl (Excel workbook) |
| Data wrangling | pandas 2.2+, NumPy 2.0+ |
| HTTP | requests 2.33+ |
| Dependency management | uv (virtual environment) |
| Deployment | Streamlit Community Cloud |

---

## 📊 Performance & Results

### Prophet Model Coverage

| Indicator | WB Code | Models Trained | Notes |
|---|---|---|---|
| GDP per Capita | NY.GDP.PCAP.KD | 15/15 | Full coverage |
| Electricity Access | EG.ELC.ACCS.ZS | 15/15 | Full coverage |
| Internet Users | IT.NET.USER.ZS | 15/15 | Full coverage |
| Life Expectancy | SP.DYN.LE00.IN | 15/15 | Full coverage |
| Under-5 Mortality | SH.DYN.MORT | 15/15 | Full coverage |
| Urbanisation | SP.URB.TOTL.IN.ZS | 15/15 | Full coverage |
| Trade Openness | NE.TRD.GNFS.ZS | 14/15 | Nigeria skipped — sparse |
| Adult Literacy Rate | SE.ADT.LITR.ZS | 5/15 | Survey-based, shown as static callout |
| **Total** | | **109** | |

### K-Means Cluster Assignments (k=3)

| Cluster | Countries |
|---|---|
| Advanced | South Africa, Egypt, Morocco, Tunisia |
| Emerging | Nigeria, Ghana, Senegal |
| Developing | Kenya, Uganda, Tanzania, Ethiopia, Rwanda, Mozambique, Zambia, Zimbabwe |

Clustering uses 5 indicators: GDP per Capita, Electricity Access, Internet Users, Life Expectancy, and Urbanisation — all StandardScaler-normalised before fitting. Cluster labels are assigned deterministically by GDP centroid rank (highest centroid = "Advanced").

Kenya sits in the **Developing** cluster alongside all EAC-5 peers (Uganda, Tanzania, Ethiopia, Rwanda). South Africa is the in-dashboard benchmark for the Advanced tier.

---

## 🌐 Live Dashboard

**https://africa-dev-forecast.streamlit.app**

The dashboard has 4 tabs:

- **10-Year Trajectory** — Select up to 15 countries and any indicator. Solid lines = historical Prophet fit (2000–2023). Dashed lines + shaded confidence band = Prophet forecast (2024–2033). Milestone markers at 2030 and 2033. Kenya KPI cards (current value → 2030 forecast → 2033 forecast with % change). Executive EAC summary panel.
- **Country Clusters** — Bubble scatter plot (GDP vs Life Expectancy; bubble size = Internet Users %; log x-axis). Kenya marked with a star. Cluster membership panels and summary metrics table.
- **Development Benchmarks** — Normalised 0–1 heatmap across 7 indicators for all 15 countries (1 = best in dataset, Under-5 Mortality inverted). Radar chart: Kenya vs Developing cluster average vs South Africa.
- **Export** — Download full Excel workbook with Historical, Clusters, and per-indicator forecast sheets.

---

## 📑 Data Sources

| Source | Description | Access |
|---|---|---|
| World Bank REST API v2 | All 8 development indicators for 15 African countries, 2000–2023 | Free, no API key |
| Endpoint | `https://api.worldbank.org/v2/country/{codes}/indicator/{code}?format=json&date=2000:2023&per_page=500` | Paginated JSON |

No data is hardcoded. Running `python data_pipeline.py` fetches fresh values from the World Bank API. Adult Literacy Rate (SE.ADT.LITR.ZS) is available from household surveys and is sparse by design — 3 to 5 observations per country over 24 years.

---

## 🧠 Key Design Decisions

### 1. Why k=3 clusters rather than k=2 or k=4

The choice of k=3 was not arbitrary. Running K-Means with k=2 through k=5 on this 15-country, 5-indicator dataset, the silhouette score peaks at k=3. More importantly, k=3 produces the only partition that is directly interpretable in the context of African development economics: North Africa plus South Africa form a clear upper tier (higher GDP, near-universal electricity, higher internet penetration, longer life expectancy); Nigeria, Ghana, and Senegal occupy a genuine middle ground (resource-driven or middle-income economies with uneven human development scores); and the EAC-plus-Southern Africa block forms a coherent lower-income, faster-improving cohort. At k=2, North Africa and South Africa are collapsed with West Africa in a single "higher" tier, which loses the analytically important distinction between them. At k=4, the Developing cluster splits arbitrarily by small GDP differences rather than by any meaningful development pattern.

### 2. Why StandardScaler is applied before K-Means

GDP per capita ranges from approximately $500 (Ethiopia, Mozambique in 2000) to over $8,000 (South Africa) in this dataset — a 16x spread. Electricity access and internet penetration are bounded percentages between 5% and 100%. Life expectancy spans roughly 47 to 76 years. Without scaling, the K-Means objective function is dominated almost entirely by the GDP dimension: two countries with identical human development profiles but a $500 GDP difference would be placed in different clusters, while a country with 95% electricity access would be treated identically to one with 20% access. StandardScaler normalises each feature to zero mean and unit variance before clustering, ensuring all five indicators contribute equally to the distance calculations. The original (unscaled) centroids are then recovered via `scaler.inverse_transform` purely for labelling clusters by GDP rank.

### 3. Why Adult Literacy Rate is shown as a static callout rather than a Prophet forecast

The World Bank publishes Adult Literacy Rate (SE.ADT.LITR.ZS) from household surveys that many countries conduct every 3 to 7 years rather than annually. Of the 15 countries in this dataset, only 5 have 8 or more non-null observations between 2000 and 2023 — the minimum the pipeline requires before training a Prophet model. For countries with 3 or 4 data points, fitting a Prophet trend with annual frequency and projecting 10 years forward would extrapolate from too few anchors and produce confidence intervals that span the entire plausible range of the indicator (e.g., 40%–100% for a country with 3 survey readings). The dashboard instead renders a static reference table showing each country's latest available literacy figure, the survey year, and the observation count, with a yellow advisory callout explaining why no forecast is available.

### 4. Why a 10-year forecast horizon

Development planning in Africa operates on decade timescales. Kenya's Vision 2030 ends in 2030; the United Nations Sustainable Development Goals have a 2030 deadline; the African Union's Agenda 2063 uses 10-year implementation plans. A 5-year horizon would not reach the 2030 milestone markers that policy analysts actually care about. A 15-year horizon would push Prophet beyond the comfortable extrapolation range for annual data with changepoints (Prophet's uncertainty bands widen substantially past 10 years when trained on only 24 historical observations). The 10-year horizon to 2033 reaches past the SDG 2030 milestone, aligns with a planning decade, and keeps the confidence intervals interpretable.

### 5. Why 15 countries rather than only the EAC-5

The EAC-5 (Kenya, Uganda, Tanzania, Ethiopia, Rwanda) are the primary countries of analytical interest for this project. However, running K-Means on 5 countries that are all at similar income levels and development stages would produce arbitrary, unstable clusters — with only 5 points, the algorithm cannot distinguish genuine development tiers from random variance in the indicators. The 15-country selection deliberately includes North African economies (Egypt, Morocco, Tunisia) and South Africa as an upper-income anchor, and Nigeria and Ghana as middle-income West African reference points. This gives K-Means enough within-group and between-group variation to produce three clusters that are geographically coherent, economically interpretable, and stable across random seeds.

---

## 📂 Project Structure

```
africa-dev-forecast/
├── app.py                          # Streamlit dashboard — 1,232 lines
├── data_pipeline.py                # World Bank fetch + Prophet + K-Means pipeline
├── requirements.txt                # Pinned dependencies
└── data/
    └── processed/                  # Generated by data_pipeline.py
        ├── development.parquet     # 360-row wide historical table
        ├── clusters.parquet        # 15-row cluster assignments
        └── forecast_<IND>_<ISO2>.parquet  # 109 Prophet forecast files
```

---

## ⚙️ Installation & Setup

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager

### Local setup

```bash
# Clone the repository
git clone https://github.com/declerke/Africa-Dev-Forecast
cd africa-dev-forecast

# Create virtual environment and install dependencies
uv venv
.venv\Scripts\activate       # Windows
# source .venv/bin/activate  # macOS/Linux

uv pip install -r requirements.txt
```

### Run the data pipeline

```bash
python data_pipeline.py
```

This fetches all 8 indicators from the World Bank API, trains 109 Prophet models, and writes parquet files to `data/processed/`. Runtime is approximately 8–12 minutes depending on network speed and Prophet fit times.

### Launch the dashboard

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

### Notes

- The pipeline requires internet access to reach the World Bank API. The dashboard runs entirely from local parquet files once the pipeline has completed.
- Prophet requires a C++ compiler for `cmdstanpy`. On Windows, install [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) if Prophet fails to install.
- `data/processed/` is committed to the repository so the dashboard works on Streamlit Community Cloud without re-running the pipeline at deploy time.

---

## 📈 Indicator Reference

| Indicator | World Bank Code | Coverage | Notes |
|---|---|---|---|
| GDP per Capita (const. 2015 USD) | NY.GDP.PCAP.KD | 15/15 countries, annual | Constant 2015 USD removes inflation effects |
| Electricity Access (%) | EG.ELC.ACCS.ZS | 15/15 countries, annual | % population with electricity access |
| Internet Users (%) | IT.NET.USER.ZS | 15/15 countries, annual | % population using the internet |
| Life Expectancy (years) | SP.DYN.LE00.IN | 15/15 countries, annual | At birth, all sexes |
| Adult Literacy Rate (%) | SE.ADT.LITR.ZS | 5/15 countries ≥8 obs | Survey-based; sparse — shown as static callout |
| Under-5 Mortality (per 1,000) | SH.DYN.MORT | 15/15 countries, annual | Deaths per 1,000 live births; lower = better |
| Urbanisation (%) | SP.URB.TOTL.IN.ZS | 15/15 countries, annual | % population in urban areas |
| Trade Openness (% GDP) | NE.TRD.GNFS.ZS | 14/15 countries | Nigeria skipped (sparse); (exports + imports) / GDP |

Adult Literacy Rate is sparse because the World Bank collects it from national household surveys (e.g., DHS, MICS) rather than annual administrative data. Most countries have 3–6 readings over 24 years. The pipeline requires a minimum of 8 non-null observations to train a Prophet model; countries below this threshold are excluded from forecasting and displayed in a static reference table instead.

---

## 🎓 Skills Demonstrated

| Skill | Evidence |
|---|---|
| REST API ingestion | Paginated World Bank v2 API with ISO3→ISO2 mapping and rate-limit delay |
| Data modelling | Long→wide pivot with enforced 15×24 grid; missing-value-aware merge |
| Time-series forecasting | Facebook Prophet with `changepoint_prior_scale=0.4`; historical vs forecast split; confidence intervals |
| Unsupervised ML | K-Means clustering with StandardScaler normalisation; centroid-based label assignment |
| Data quality handling | Sparse-indicator detection; minimum-observation guard (`MIN_OBS_FOR_PROPHET = 8`); graceful skip with pipeline logging |
| Dashboard engineering | 4-tab Streamlit app; `@st.cache_data` on all loaders; custom CSS injection (DM Sans + Space Grotesk; dark theme) |
| Plotly visualisation | Forecast chart with confidence bands; bubble scatter (log x-axis, size-encoded); normalised heatmap; radar chart |
| Parquet-based pipeline | Decoupled pipeline and dashboard; 111 parquet files; PyArrow columnar storage |
| Excel export | Multi-sheet openpyxl workbook (Historical + Clusters + per-indicator forecast sheets) |
| Production deployment | Streamlit Community Cloud deployment; committed processed data for zero-pipeline-runtime serving |
