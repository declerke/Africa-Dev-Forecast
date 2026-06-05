"""
Africa Development Trajectory Forecaster — Data Pipeline
Fetches World Bank indicators, trains Prophet models, clusters countries.
"""

import os
import time
import warnings
import logging

import numpy as np
import pandas as pd
import requests
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
logging.getLogger("prophet").setLevel(logging.ERROR)

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

WB_BASE = "https://api.worldbank.org/v2"
DATE_RANGE = "2000:2023"
PROCESSED_DIR = os.path.join("data", "processed")
MIN_OBS_FOR_PROPHET = 8
FORECAST_YEARS = 10

os.makedirs(PROCESSED_DIR, exist_ok=True)


# ── World Bank Fetch ─────────────────────────────────────────────────────────

def fetch_indicator(indicator_code: str, country_codes: list[str]) -> pd.DataFrame:
    """Fetch a single indicator for all countries via World Bank REST API."""
    codes = ";".join(country_codes)
    url = (
        f"{WB_BASE}/country/{codes}/indicator/{indicator_code}"
        f"?format=json&date={DATE_RANGE}&per_page=500"
    )
    rows = []
    page = 1
    while True:
        paged_url = f"{url}&page={page}"
        try:
            resp = requests.get(paged_url, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"  [WARN] fetch failed for {indicator_code} page {page}: {e}")
            break

        if not isinstance(data, list) or len(data) < 2:
            break

        meta, records = data[0], data[1]
        if records is None:
            break

        for rec in records:
            rows.append(
                {
                    "country_code": rec["countryiso3code"]
                    if rec.get("countryiso3code")
                    else (rec["country"]["id"] if rec.get("country") else None),
                    "country": rec["country"]["value"] if rec.get("country") else None,
                    "year": int(rec["date"]),
                    "value": float(rec["value"]) if rec["value"] is not None else None,
                    "indicator_code": indicator_code,
                    "indicator_label": INDICATORS[indicator_code],
                }
            )

        total_pages = meta.get("pages", 1)
        if page >= total_pages:
            break
        page += 1
        time.sleep(0.2)

    df = pd.DataFrame(rows)
    if not df.empty and "country_code" in df.columns:
        # World Bank returns ISO3; map back to ISO2 for consistency
        iso3_to_iso2 = {
            "KEN": "KE", "UGA": "UG", "TZA": "TZ", "ETH": "ET", "RWA": "RW",
            "NGA": "NG", "GHA": "GH", "SEN": "SN", "ZAF": "ZA", "EGY": "EG",
            "MAR": "MA", "TUN": "TN", "MOZ": "MZ", "ZMB": "ZM", "ZWE": "ZW",
        }
        df["iso2"] = df["country_code"].map(iso3_to_iso2)
        df = df[df["iso2"].notna()].copy()
    return df


def fetch_all_indicators() -> pd.DataFrame:
    """Fetch all indicators for all countries and return combined DataFrame."""
    country_codes = list(COUNTRIES.keys())
    all_frames = []

    for code, label in INDICATORS.items():
        print(f"  Fetching: {label} ({code})...")
        df = fetch_indicator(code, country_codes)
        if not df.empty:
            all_frames.append(df)
            print(f"    -> {len(df)} rows")
        else:
            print(f"    -> No data returned")

    if not all_frames:
        raise RuntimeError("No data fetched from World Bank API.")

    combined = pd.concat(all_frames, ignore_index=True)
    return combined


# ── Pivot & Save ─────────────────────────────────────────────────────────────

def build_wide_table(long_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot long→wide and create full grid of 15 countries × 24 years."""
    pivot = long_df.pivot_table(
        index=["iso2", "year"],
        columns="indicator_code",
        values="value",
        aggfunc="mean",
    ).reset_index()

    # Ensure full grid: 15 countries × years 2000-2023
    countries = list(COUNTRIES.keys())
    years = list(range(2000, 2024))
    grid = pd.MultiIndex.from_product([countries, years], names=["iso2", "year"])
    grid_df = pd.DataFrame(index=grid).reset_index()

    wide = grid_df.merge(pivot, on=["iso2", "year"], how="left")
    wide["economy"] = wide["iso2"].map(COUNTRIES)

    # Reorder columns
    ind_cols = [c for c in INDICATORS.keys() if c in wide.columns]
    wide = wide[["iso2", "economy", "year"] + ind_cols]
    return wide


# ── Prophet Forecasting ──────────────────────────────────────────────────────

def train_prophet_models(long_df: pd.DataFrame) -> dict:
    """
    Train a Prophet model per (country, indicator) pair with ≥ MIN_OBS_FOR_PROPHET.
    Returns a dict of coverage stats per indicator.
    """
    try:
        from prophet import Prophet
    except ImportError:
        raise ImportError("prophet not installed")

    stats = {}
    total_trained = 0

    for ind_code, ind_label in INDICATORS.items():
        models_trained = 0
        countries_skipped = []

        subset = long_df[long_df["indicator_code"] == ind_code].copy()
        subset = subset[subset["value"].notna()].copy()

        for iso2, economy in COUNTRIES.items():
            country_data = subset[subset["iso2"] == iso2].sort_values("year")

            if len(country_data) < MIN_OBS_FOR_PROPHET:
                countries_skipped.append(iso2)
                continue

            # Prophet requires ds (datetime) + y (float)
            prophet_df = pd.DataFrame(
                {
                    "ds": pd.to_datetime(
                        country_data["year"].astype(str) + "-01-01"
                    ),
                    "y": country_data["value"].values,
                }
            )

            try:
                model = Prophet(
                    changepoint_prior_scale=0.4,
                    yearly_seasonality=False,
                    weekly_seasonality=False,
                    daily_seasonality=False,
                )
                model.fit(prophet_df)

                future = model.make_future_dataframe(
                    periods=FORECAST_YEARS, freq="YS"
                )
                forecast = model.predict(future)

                # Merge with original to mark historical vs forecast
                forecast_df = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
                cutoff = pd.Timestamp(f"{prophet_df['ds'].dt.year.max() + 1}-01-01")
                forecast_df["is_forecast"] = forecast_df["ds"] >= cutoff

                safe_ind = ind_code.replace(".", "_")
                out_path = os.path.join(
                    PROCESSED_DIR, f"forecast_{safe_ind}_{iso2}.parquet"
                )
                forecast_df.to_parquet(out_path, index=False)
                models_trained += 1
                total_trained += 1

            except Exception as e:
                print(f"    [WARN] Prophet failed for {iso2}/{ind_code}: {e}")
                countries_skipped.append(iso2)

        stats[ind_code] = {
            "label": ind_label,
            "models_trained": models_trained,
            "skipped": countries_skipped,
        }
        skip_str = f", skipped: {countries_skipped}" if countries_skipped else ""
        print(
            f"  {ind_label}: {models_trained} models trained{skip_str}"
        )

    print(f"\n  Total Prophet models trained: {total_trained}")
    return stats


# ── K-Means Clustering ───────────────────────────────────────────────────────

def cluster_countries(wide_df: pd.DataFrame) -> pd.DataFrame:
    """
    K-Means(k=3) on 2023 (or latest) values of 5 indicators.
    Labels clusters by GDP per capita centroid ranking.
    """
    CLUSTER_FEATURES = [
        "NY.GDP.PCAP.KD",
        "EG.ELC.ACCS.ZS",
        "IT.NET.USER.ZS",
        "SP.DYN.LE00.IN",
        "SP.URB.TOTL.IN.ZS",
    ]

    # Get latest non-null value per country for each feature
    records = []
    for iso2, economy in COUNTRIES.items():
        row = {"iso2": iso2, "economy": economy}
        cdf = wide_df[wide_df["iso2"] == iso2].sort_values("year", ascending=False)
        for feat in CLUSTER_FEATURES:
            if feat in cdf.columns:
                val_series = cdf[feat].dropna()
                row[feat] = val_series.iloc[0] if not val_series.empty else None
            else:
                row[feat] = None
        records.append(row)

    cluster_df = pd.DataFrame(records)
    feature_matrix = cluster_df[CLUSTER_FEATURES].values.astype(float)

    # Impute then scale
    imputer = SimpleImputer(strategy="mean")
    feature_imp = imputer.fit_transform(feature_matrix)

    scaler = StandardScaler()
    feature_scaled = scaler.fit_transform(feature_imp)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    cluster_df["cluster_id"] = kmeans.fit_predict(feature_scaled)

    # Label clusters by GDP centroid rank (un-scaled centroid)
    centroids_scaled = kmeans.cluster_centers_
    centroids_orig = scaler.inverse_transform(centroids_scaled)
    gdp_col_idx = CLUSTER_FEATURES.index("NY.GDP.PCAP.KD")
    gdp_centroids = centroids_orig[:, gdp_col_idx]

    rank_order = np.argsort(gdp_centroids)[::-1]  # descending
    label_map = {
        rank_order[0]: "Advanced",
        rank_order[1]: "Emerging",
        rank_order[2]: "Developing",
    }
    cluster_df["cluster_label"] = cluster_df["cluster_id"].map(label_map)

    out_path = os.path.join(PROCESSED_DIR, "clusters.parquet")
    cluster_df.to_parquet(out_path, index=False)
    return cluster_df


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Africa Development Trajectory Forecaster — Data Pipeline")
    print("=" * 60)

    # 1. Fetch all data
    print("\n[1/4] Fetching World Bank indicators...")
    long_df = fetch_all_indicators()
    print(f"\n  Total raw rows fetched: {len(long_df)}")

    # 2. Build wide table and save
    print("\n[2/4] Building processed dataset...")
    wide_df = build_wide_table(long_df)
    out_path = os.path.join(PROCESSED_DIR, "development.parquet")
    wide_df.to_parquet(out_path, index=False)
    print(f"  Saved {len(wide_df)} rows → {out_path}")
    print(
        f"  Expected base grid: {len(COUNTRIES)} countries × 24 years = "
        f"{len(COUNTRIES) * 24} rows"
    )

    # 3. Train Prophet models
    print("\n[3/4] Training Prophet forecasting models...")
    forecast_stats = train_prophet_models(long_df)

    # 4. Cluster countries
    print("\n[4/4] Clustering countries (K-Means, k=3)...")
    cluster_df = cluster_countries(wide_df)

    print("\n--- Cluster Assignments ---")
    for _, row in cluster_df.sort_values("cluster_label").iterrows():
        print(
            f"  {row['iso2']} ({row['economy']:20s}) → "
            f"Cluster {row['cluster_id']}: {row['cluster_label']}"
        )

    print("\n--- Indicator Coverage Summary ---")
    for code, stat in forecast_stats.items():
        coverage = stat["models_trained"]
        total = len(COUNTRIES)
        flag = "[sparse] " if code == "SE.ADT.LITR.ZS" else "  "
        print(
            f"{flag}{stat['label']}: "
            f"{coverage}/{total} countries forecasted"
            + (
                f"  [sparse: {stat['skipped']}]"
                if stat["skipped"]
                else ""
            )
        )

    print("\n[Done] Pipeline complete.")


if __name__ == "__main__":
    main()
