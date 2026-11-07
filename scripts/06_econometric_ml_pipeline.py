#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
06_econometric_ml_pipeline.py
=============================
Unified Econometric (Progressive OLS & Robustness) and Spatial Machine Learning
(GroupKFold CV, TreeSHAP, and Partial Dependence Thresholds) Pipeline.
Reproduces all empirical tables in Section 5 and the Appendix (Outputs_v8):
  - Table 1: Metropolitan Descriptive Statistics & Polycentric/CLIP Medians
  - Table 2: Progressive 5-Stage Hedonic OLS Sequence (Models 1 - 5) with Colonia-Clustered SEs
  - Table 3: Random 5-Fold CV vs. Spatial-Block GroupKFold CV Benchmarks
  - Table 4: TreeSHAP Global Feature Attributions & Directional Correlations
  - Table 5: 1D Partial Dependence Swings & Primary Inflection Thresholds (x_S*)
  - Appendix Table A1: Robustness Checks (Exact Matches, Real Photos, <=10km Core, Colonia FE)
  - Appendix Table A2: Metropolitan Submarket Hedonic Regressions

"""

import os
import re
import glob
import datetime
import warnings
import unicodedata
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import scipy.stats as stats

import statsmodels.api as sm
import statsmodels.formula.api as smf

from sklearn.model_selection import KFold, GroupKFold
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.inspection import partial_dependence, permutation_importance

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image, ImageDraw, ImageOps
import seaborn as sns

warnings.filterwarnings("ignore")

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


# ==============================================================================
# 1. CONFIGURATION, PATHS & DOMAIN COLOR TAXONOMY
# ==============================================================================
DEFAULT_BASE_DIR = r"C:\Users\multiple_cities_v3"
FALLBACK_BASE_DIR = r"C:\Users\multiple_cities_v2"

BASE_DIR = DEFAULT_BASE_DIR if os.path.exists(DEFAULT_BASE_DIR) else FALLBACK_BASE_DIR
INPUTS_DIR = os.path.join(BASE_DIR, "Inputs")
DATA_DIR = os.path.join(BASE_DIR, "Data")
CITY_PHOTOS_DIR = os.path.join(INPUTS_DIR, "CityPhotos")

OUTPUTS_DIR = os.path.join(BASE_DIR, "Outputs_v8")
TABLES_DIR = os.path.join(OUTPUTS_DIR, "Tables")
FIGURES_DIR = os.path.join(OUTPUTS_DIR, "Figures")

for d in [INPUTS_DIR, DATA_DIR, CITY_PHOTOS_DIR, OUTPUTS_DIR, TABLES_DIR, FIGURES_DIR]:
    os.makedirs(d, exist_ok=True)

METRIC_CRS = "EPSG:6372"
RANDOM_SEED = 42
MAX_CORE_DIST_KM = 15.0

METRO_CENTERS = {
    "Queretaro": (20.5931, -100.3921),           # Jardín Zenea, Querétaro
    "Aguascalientes": (21.8810, -102.2963),      # Plaza de la Patria, Aguascalientes
    "Leon": (21.1220, -101.6825),                # Plaza de los Mártires, León
    "Zacatecas-Guadalupe": (22.7764, -102.5720)  # Plaza de Armas, Zacatecas
}

METRO_DISPLAY_NAMES = {
    "Queretaro": "Querétaro (Jardín Zenea)",
    "Aguascalientes": "Aguascalientes (Plaza de la Patria)",
    "Leon": "León (Plaza de los Mártires)",
    "Zacatecas-Guadalupe": "Zacatecas-Guadalupe (Plaza de Armas)"
}

METRO_LATEX_NAMES = {
    "Queretaro": r"Quer\'etaro",
    "Aguascalientes": "Aguascalientes",
    "Leon": r"Le\'on",
    "Zacatecas-Guadalupe": "Zacatecas-Guadalupe"
}

TARGET_MUN_TO_METRO = {
    "01001": "Aguascalientes", "01005": "Aguascalientes", "01011": "Aguascalientes",
    "11020": "Leon", "11025": "Leon", "11031": "Leon",
    "22014": "Queretaro", "22006": "Queretaro", "22011": "Queretaro", "22008": "Queretaro",
    "32056": "Zacatecas-Guadalupe", "32017": "Zacatecas-Guadalupe",
    "32032": "Zacatecas-Guadalupe", "32050": "Zacatecas-Guadalupe", "32057": "Zacatecas-Guadalupe"
}

DOMAIN_PALETTE = {
    "Structural Scale": "#1f4e78",         # Deep Navy
    "Polycentric & Enclave": "#008080",    # Teal
    "Neighborhood & DENUE": "#6c3483",     # Royal Purple
    "Computer Vision (CLIP)": "#d35400",   # Burnt Coral
    "Submarket Controls": "#566573"        # Slate Gray
}

FEATURE_META = {
    "ln_built_area": ("Log Built Floor Area (ln m²)", r"Log Built Floor Area ($\ln A_{icm}$)", "Structural Scale"),
    "bedrooms": ("Bedrooms (Count)", r"Bedrooms ($B_{icm}$)", "Structural Scale"),
    "bathrooms": ("Bathrooms (Count)", r"Bathrooms ($H_{icm}$)", "Structural Scale"),
    "dist_to_historic_center_km": ("Distance to Historic CBD (km)", r"Distance to Historic CBD ($D^{\text{CBD}}_{cm}$)", "Polycentric & Enclave"),
    "dist_retail_subcenter_km": ("Distance to Retail Subcenter (km)", r"Distance to Retail Subcenter ($D^{\text{RET}}_{cm}$)", "Polycentric & Enclave"),
    "dist_industrial_subcenter_km": ("Distance to Industrial Subcenter (km)", r"Distance to Industrial Subcenter ($D^{\text{IND}}_{cm}$)", "Polycentric & Enclave"),
    "dist_industrial_subcenter_km_sq": ("Industrial Distance Squared (km²)", r"Industrial Distance Squared ($(D^{\text{IND}}_{cm})^2$)", "Polycentric & Enclave"),
    "is_planned_gated_enclave": ("Planned Gated Enclave (0/1)", r"Planned Gated Enclave ($G_{icm}$)", "Polycentric & Enclave"),
    "is_planned_gated_enclave:dist_to_historic_center_km": ("Gated Enclave × Distance to CBD", r"Gated $\times$ Distance to CBD ($G_{icm} \times D^{\text{CBD}}_{cm}$)", "Polycentric & Enclave"),
    "amenity_index_1km": ("Log DENUE Amenity Density (1km)", r"Log Positive Amenity Density ($I_{cm}$)", "Neighborhood & DENUE"),
    "nightlife_index_1km": ("Log Nightlife Disamenity Density (1km)", r"Log Nightlife Disamenity ($N_{cm}$)", "Neighborhood & DENUE"),
    "z_marginalization": ("Urban Deprivation Z-Score (CONAPO)", r"Urban Deprivation Z-Score ($Z_{\text{marg},cm}$)", "Neighborhood & DENUE"),
    "imc_pct_no_basic_educ": ("Incomplete Basic Education Share (%)", r"Incomplete Basic Education ($E_{cm}$)", "Neighborhood & DENUE"),
    "census_internet_share": ("Household Internet Access Share", r"Household Internet Share ($W_{cm}$)", "Neighborhood & DENUE"),
    "cv_has_cgi_render": ("Presale CGI Render Indicator (CLIP)", r"Presale CGI Render Indicator ($R_{icm}$)", "Computer Vision (CLIP)"),
    "cv_share_interior": ("Interior Room Photo Share (CLIP)", r"Interior Room Photo Share ($S_{icm}$)", "Computer Vision (CLIP)"),
    "cv_scene_diversity_entropy": ("Gallery Scene Diversity Entropy", r"Scene Diversity Entropy ($H_{icm}^{\text{scene}}$)", "Computer Vision (CLIP)"),
    "cv_prob_good_condition_mean": ("CLIP Structural Conservation Score", r"Structural Conservation Score ($C_{icm}$)", "Computer Vision (CLIP)"),
    "cv_prob_greenery_max": ("CLIP Streetscape Greenery Score", r"Streetscape Greenery Score ($V_{icm}$)", "Computer Vision (CLIP)"),
    "cv_modern_int_clean": ("CLIP Modern Interior Finishes Score", r"Modern Interior Finishes Score ($F_{icm}$)", "Computer Vision (CLIP)"),
    "cv_brightness_mean": ("Photometric Luminance (Mean)", r"Photometric Luminance ($L_{icm}$)", "Computer Vision (CLIP)"),
    "ln_sharpness": ("Log Laplacian Image Sharpness", r"Log Laplacian Sharpness ($Z_{icm}$)", "Computer Vision (CLIP)"),
    "metro_Queretaro": ("Submarket: Querétaro", r"Submarket: Quer\'etaro", "Submarket Controls"),
    "metro_Aguascalientes": ("Submarket: Aguascalientes", r"Submarket: Aguascalientes", "Submarket Controls"),
    "metro_Leon": ("Submarket: León", r"Submarket: Le\'on", "Submarket Controls"),
    "metro_Zacatecas-Guadalupe": ("Submarket: Zacatecas-Guadalupe", r"Submarket: Zacatecas-Guadalupe", "Submarket Controls")
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.8
})


# ==============================================================================
# 2. DATA LOADER: MERGES SPATIAL,  SUBCENTERS  & CLIP CV FEATURES
# ==============================================================================
def load_conapo_polygons_for_map():
    shp_candidates = [
        os.path.join(DATA_DIR, "colonias_imc2020_shp", "colonias_imc2020.shp"),
        os.path.join(INPUTS_DIR, "colonias_imc2020_shp", "colonias_imc2020.shp")
    ]
    shp_path = next((p for p in shp_candidates if os.path.exists(p)), None)
    if not shp_path:
        return None
    for enc in (None, "utf-8", "cp1252", "latin1"):
        try:
            gdf = gpd.read_file(shp_path, encoding=enc) if enc else gpd.read_file(shp_path)
            break
        except Exception:
            continue
    if gdf is None or gdf.empty:
        return None
    if gdf.crs is None:
        gdf.set_crs(METRIC_CRS, inplace=True)
    else:
        gdf = gdf.to_crs(METRIC_CRS)

    id_col = "CVE_COL" if "CVE_COL" in gdf.columns else "CVEGEO"
    gdf["cve_col_9"] = gdf[id_col].astype(str).str.replace(r"\.0+$", "", regex=True).str.replace("_", "", regex=False).str.zfill(9)
    gdf["cve_mun_5"] = gdf["cve_col_9"].str.slice(0, 5)
    gdf = gdf[gdf["cve_mun_5"].isin(TARGET_MUN_TO_METRO.keys())].copy()
    gdf["metro_shp"] = gdf["cve_mun_5"].map(TARGET_MUN_TO_METRO)

    tipo_col = "CLASIF" if "CLASIF" in gdf.columns else "TIPO"
    nom_col = "COLONIA" if "COLONIA" in gdf.columns else "NOM_ASEN"
    comb = (gdf.get(tipo_col, pd.Series("", index=gdf.index)).astype(str) + " " + gdf.get(nom_col, pd.Series("", index=gdf.index)).astype(str)).str.upper()
    gdf["is_gated_Poly"] = comb.str.contains("FRACCIONAMIENTO|PRIVADA|RESIDENCIAL|CONDOMINIO|COTO|CONJUNTO HABITACIONAL|CLUB DE GOLF", regex=True).astype(int)
    return gdf


def load_v3_dataset_and_subcenters():
    print("[1/6] Loading 15km DENUE Spatial Dataset (v3) and CLIP Computer Vision Features...")
    spatial_v3_path = os.path.join(INPUTS_DIR, "property_listings_multicity_filtered_15km_v3.csv")
    if not os.path.exists(spatial_v3_path):
        raise FileNotFoundError(
            f"{spatial_v3_path} not found. Run `04_spatial_features_v3.py` first."
        )

    df = pd.read_csv(spatial_v3_path, low_memory=False)
    df["listing_id"] = df["listing_id"].astype(str).str.strip()
    df["colonia_cvegeo"] = df["cve_col_9"].astype(str).str.zfill(9)

    # Locate and merge CLIP Computer Vision features (cv_features_hedonic.csv)
    cv_candidates = [
        os.path.join(INPUTS_DIR, "cv_features_hedonic.csv"),
        os.path.join(DATA_DIR, "cv_features_hedonic.csv"),
        os.path.join(BASE_DIR, "CV_Outputs_v2", "cv_features_hedonic.csv"),
        os.path.join(BASE_DIR, "CV_Outputs", "cv_features_hedonic.csv"),
        os.path.join(FALLBACK_BASE_DIR, "Inputs", "cv_features_hedonic.csv"),
        os.path.join(FALLBACK_BASE_DIR, "Data", "cv_features_hedonic.csv"),
        os.path.join(FALLBACK_BASE_DIR, "CV_Outputs_v2", "cv_features_hedonic.csv")
    ]
    cv_path = next((p for p in cv_candidates if os.path.exists(p) and os.path.getsize(p) > 300), None)

    if cv_path:
        print(f"      -> Merging CLIP Computer Vision features from: {cv_path}")
        df_cv = pd.read_csv(cv_path, dtype={"listing_id": str})
        df_cv["listing_id"] = df_cv["listing_id"].astype(str).str.strip()
        df_cv.drop_duplicates(subset=["listing_id"], keep="last", inplace=True)
        cv_cols = [c for c in df_cv.columns if c != "listing_id"]
        df.drop(columns=[c for c in cv_cols if c in df.columns], inplace=True, errors="ignore")
        df = pd.merge(df, df_cv, on="listing_id", how="left")
    else:
        print("      [WARN] cv_features_hedonic.csv not found in standard paths; checking existing CV columns in df.")

    # Ensure numeric types on core variables
    num_cols = [
        "price", "built_area_m2", "price_m2", "ln_price", "ln_price_m2", "ln_built_area",
        "bedrooms", "bathrooms", "dist_to_historic_center_km", "dist_retail_subcenter_km",
        "dist_industrial_subcenter_km", "dist_industrial_subcenter_km_sq",
        "is_gated_polygon", "is_gated_listing", "is_planned_gated_enclave",
        "amenity_index_1km", "nightlife_index_1km", "industrial_buffer_index_1km",
        "imc_marginalization_index", "z_marginalization", "imc_pct_no_basic_educ",
        "census_internet_share", "x_metric", "y_metric"
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in [
        "cv_has_cgi_render", "cv_share_interior", "cv_has_interior_photo",
        "cv_scene_diversity_entropy", "cv_prob_good_condition_mean",
        "cv_prob_greenery_max", "cv_prob_modern_finishes_max",
        "cv_prob_modern_interior_only", "cv_brightness_mean", "cv_sharpness_mean"
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "cv_has_interior_photo" not in df.columns:
        df["cv_has_interior_photo"] = (df["cv_share_interior"].fillna(0) > 0).astype(int)

    
    raw_mod_int = df["cv_prob_modern_interior_only"].copy()
    has_int_mask = df["cv_has_interior_photo"] == 1
    raw_mod_int.loc[~has_int_mask] = np.nan

    # Impute missing visual scores using (metro_area, is_planned_gated_enclave) group medians
    # rather than a single global constant so cross-ring and cross-city variation is preserved
    df["cv_modern_int_clean"] = raw_mod_int.fillna(
        df.groupby(["metro_area", "is_planned_gated_enclave"])["cv_prob_modern_interior_only"].transform(
            lambda s: s[df.loc[s.index, "cv_has_interior_photo"] == 1].median()
        )
    ).fillna(raw_mod_int.median())

    for cv_c in ["cv_prob_good_condition_mean", "cv_prob_greenery_max", "cv_share_interior", "cv_scene_diversity_entropy", "cv_brightness_mean", "cv_sharpness_mean"]:
        if cv_c in df.columns:
            df[cv_c] = df[cv_c].fillna(
                df.groupby(["metro_area", "is_planned_gated_enclave"])[cv_c].transform("median")
            ).fillna(df[cv_c].median())

    df["cv_has_cgi_render"] = df["cv_has_cgi_render"].fillna(0).astype(int)
    df["ln_sharpness"] = np.log1p(np.clip(df["cv_sharpness_mean"], 0.0, None))

    # Build Spatial Blocks for GroupKFold Cross-Validation (5 Contiguous Sectors per City)
    df["spatial_block_id"] = ""
    for m in df["metro_area"].unique():
        sub_idx = df[df["metro_area"] == m].index
        sub_coords = df.loc[sub_idx, ["x_metric", "y_metric"]].values
        km_blk = KMeans(n_clusters=5, random_state=RANDOM_SEED, n_init=15).fit(sub_coords)
        df.loc[sub_idx, "spatial_block_id"] = [f"{m}_BLK_{b}" for b in km_blk.labels_]

    # Load Exogenous DENUE Subcenters for Cartography
    sub_csv_path = os.path.join(INPUTS_DIR, "polycentric_subcenters_v3.csv")
    subcenters_df = pd.read_csv(sub_csv_path)
    subcenters_dict = {}
    for m in METRO_CENTERS.keys():
        m_sub = subcenters_df[subcenters_df["metro_area"] == m]
        cbd_xy = m_sub[m_sub["subcenter_type"] == "cbd"][["x_metric", "y_metric"]].values[0]
        ret_xy = m_sub[m_sub["subcenter_type"] == "retail"][["x_metric", "y_metric"]].values
        ind_xy = m_sub[m_sub["subcenter_type"] == "industrial"][["x_metric", "y_metric"]].values
        subcenters_dict[m] = {"cbd": (cbd_xy[0], cbd_xy[1]), "retail": ret_xy, "industrial": ind_xy}

    gdf_polygons = load_conapo_polygons_for_map()
    print(f"      -> Dataset ready: N = {len(df):,} listings across {df['colonia_cvegeo'].nunique()} colonias.")
    return df, gdf_polygons, subcenters_dict


# ==============================================================================
# 3. 5-MODEL PROGRESSIVE BUNDLING OLS & WITHIN-COLONIA FE ROBUSTNESS
# ==============================================================================
def run_hedonic_ols_models(df):
    print("[2/6] Estimating 5-Model Morphological & Visual Bundling OLS Sequence...")

    # Model 1: Polycentric Spatial Baseline (Tests pure spatial & quadratic industrial gradients)
    f_m1 = (
        "ln_price ~ dist_to_historic_center_km + dist_retail_subcenter_km + "
        "dist_industrial_subcenter_km + dist_industrial_subcenter_km_sq + C(metro_area)"
    )

    # Model 2: + Gated Enclave Morphology & Peripheral Inversion Interaction
    f_m2 = (
        f_m1 + " + is_planned_gated_enclave + is_planned_gated_enclave:dist_to_historic_center_km"
    )

    # Model 3: + Neighborhood Socioeconomic Deprivation & DENUE 1km Buffers
    f_m3 = (
        f_m2 + " + z_marginalization + census_internet_share + amenity_index_1km + nightlife_index_1km"
    )

    # Model 4: + Structural Scale Bundling (Tests how floor area & bathrooms mediate the enclave premium)
    f_m4 = (
        f_m3 + " + ln_built_area + bedrooms + bathrooms"
    )

    # Model 5: + Zero-Shot CLIP Computer Vision & Presale Signaling (Full Multimodal Model)
    f_m5 = (
        f_m4 + " + cv_has_cgi_render + cv_share_interior + "
        "cv_prob_good_condition_mean + cv_prob_greenery_max + cv_modern_int_clean + ln_sharpness"
    )

    models = {}
    for name, formula in [
        ("Model 1 (Polycentric)", f_m1),
        ("Model 2 (+Gated Enclave)", f_m2),
        ("Model 3 (+Neighborhood)", f_m3),
        ("Model 4 (+Structural)", f_m4),
        ("Model 5 (+Computer Vision)", f_m5)
    ]:
        mod = smf.ols(formula=formula, data=df).fit(
            cov_type="cluster", cov_kwds={"groups": df["colonia_cvegeo"]}
        )
        models[name] = mod

    # Metropolitan Submarket Regressions (Appendix Table A2)
    f_city = (
        "ln_price ~ ln_built_area + bedrooms + bathrooms + "
        "dist_to_historic_center_km + dist_retail_subcenter_km + dist_industrial_subcenter_km + "
        "is_planned_gated_enclave + amenity_index_1km + nightlife_index_1km + z_marginalization + "
        "cv_has_cgi_render + cv_share_interior + cv_prob_good_condition_mean + "
        "cv_prob_greenery_max + cv_modern_int_clean"
    )
    city_models = {}
    for m in ["Queretaro", "Aguascalientes", "Leon", "Zacatecas-Guadalupe"]:
        sub = df[df["metro_area"] == m].copy()
        city_models[m] = smf.ols(formula=f_city, data=sub).fit(
            cov_type="cluster", cov_kwds={"groups": sub["colonia_cvegeo"]}
        )

    # Appendix Table A1: Econometric Robustness Checks 
    robustness_models = {}

    # R1: Exact Normalized Colonia Name Matches Only (Excludes Fuzzy Matches)
    df_exact = df[df["colonia_match_method"] == "exact_normalized_name"].copy()
    print(f"      -> Robustness R1 (Exact Colonia Matches): N = {len(df_exact):,} (dropped {len(df) - len(df_exact)} fuzzy matches)")
    robustness_models["R1: Exact Matches"] = smf.ols(formula=f_m5, data=df_exact).fit(
        cov_type="cluster", cov_kwds={"groups": df_exact["colonia_cvegeo"]}
    )

    # R2: Authentic Camera Photographs Only (Excludes Presale CGI Renders, R_icm == 0)
    df_real = df[df["cv_has_cgi_render"] == 0].copy()
    f_m5_no_cgi = f_m5.replace(" + cv_has_cgi_render", "")
    robustness_models["R2: Real Photos Only"] = smf.ols(formula=f_m5_no_cgi, data=df_real).fit(
        cov_type="cluster", cov_kwds={"groups": df_real["colonia_cvegeo"]}
    )

    # R3: Tightened Inner Core Boundary (<= 10.0 km from Historic CBD)
    df_10km = df[df["dist_to_historic_center_km"] <= 10.0].copy()
    robustness_models["R3: Core <= 10 km"] = smf.ols(formula=f_m5, data=df_10km).fit(
        cov_type="cluster", cov_kwds={"groups": df_10km["colonia_cvegeo"]}
    )

    # R4: Within-Colonia Fixed Effects (572 Colonia Fixed Effects absorbing all polygon unobservables!)
    f_within_fe = (
        "ln_price ~ ln_built_area + bedrooms + bathrooms + is_planned_gated_enclave + "
        "cv_has_cgi_render + cv_share_interior + cv_prob_good_condition_mean + "
        "cv_prob_greenery_max + cv_modern_int_clean + ln_sharpness + C(colonia_cvegeo)"
    )
    robustness_models["R4: Colonia Fixed Effects"] = smf.ols(formula=f_within_fe, data=df).fit(
        cov_type="cluster", cov_kwds={"groups": df["colonia_cvegeo"]}
    )

    return models, city_models, robustness_models


# ==============================================================================
# 4. SPATIAL-BLOCK vs. RANDOM CROSS-VALIDATION & TREESHAP ENGINE
# ==============================================================================
ML_FEATURE_COLS = [
    "ln_built_area", "bedrooms", "bathrooms",
    "dist_to_historic_center_km", "dist_retail_subcenter_km", "dist_industrial_subcenter_km",
    "is_planned_gated_enclave", "amenity_index_1km", "nightlife_index_1km",
    "z_marginalization", "imc_pct_no_basic_educ", "census_internet_share",
    "cv_has_cgi_render", "cv_share_interior", "cv_scene_diversity_entropy",
    "cv_prob_good_condition_mean", "cv_prob_greenery_max", "cv_modern_int_clean",
    "cv_brightness_mean", "ln_sharpness"
]


def prepare_ml_matrices(df):
    df_ml = df.copy()
    metro_dummies = pd.get_dummies(df_ml["metro_area"], prefix="metro", drop_first=False, dtype=float)
    X = pd.concat([df_ml[ML_FEATURE_COLS].astype(float), metro_dummies], axis=1)
    X = X.fillna(X.median()).fillna(0.0)
    y = df_ml["ln_price"].values
    groups_spatial = df_ml["spatial_block_id"].values
    groups_colonia = df_ml["colonia_cvegeo"].values
    return X, y, groups_spatial, groups_colonia


def evaluate_models_cv(df):
    print("[3/6] Benchmarking Econometric vs. Tree Ensemble models under Random & Spatial-Block CV...")
    X, y, groups_spatial, _ = prepare_ml_matrices(df)

    estimators = {
        "Hedonic OLS": Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("reg", LinearRegression())]),
        "Ridge Regression (L2)": Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("reg", Ridge(alpha=10.0, random_state=RANDOM_SEED))]),
        "Random Forest": RandomForestRegressor(n_estimators=300, max_depth=12, min_samples_leaf=6, random_state=RANDOM_SEED, n_jobs=-1),
        "HistGBDT Ensemble": HistGradientBoostingRegressor(max_iter=220, learning_rate=0.04, max_leaf_nodes=23, min_samples_leaf=15, l2_regularization=2.5, random_state=RANDOM_SEED)
    }

    kf_random = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    gkf_spatial = GroupKFold(n_splits=5)

    cv_records = []
    for model_name, est in estimators.items():
        r2_rand, rmse_rand, mae_rand = [], [], []
        for tr_idx, te_idx in kf_random.split(X, y):
            est.fit(X.iloc[tr_idx], y[tr_idx])
            preds = est.predict(X.iloc[te_idx])
            r2_rand.append(r2_score(y[te_idx], preds))
            rmse_rand.append(np.sqrt(mean_squared_error(y[te_idx], preds)))
            mae_rand.append(mean_absolute_error(y[te_idx], preds))

        r2_spat, rmse_spat, mae_spat = [], [], []
        for tr_idx, te_idx in gkf_spatial.split(X, y, groups=groups_spatial):
            est.fit(X.iloc[tr_idx], y[tr_idx])
            preds = est.predict(X.iloc[te_idx])
            r2_spat.append(r2_score(y[te_idx], preds))
            rmse_spat.append(np.sqrt(mean_squared_error(y[te_idx], preds)))
            mae_spat.append(mean_absolute_error(y[te_idx], preds))

        cv_records.append({
            "Model": model_name,
            "Random_CV_R2": np.mean(r2_rand),
            "Random_CV_R2_SD": np.std(r2_rand),
            "Random_CV_RMSE": np.mean(rmse_rand),
            "Random_CV_MAE": np.mean(mae_rand),
            "Spatial_CV_R2": np.mean(r2_spat),
            "Spatial_CV_R2_SD": np.std(r2_spat),
            "Spatial_CV_RMSE": np.mean(rmse_spat),
            "Spatial_CV_MAE": np.mean(mae_spat),
            "Generalization_Gap_R2": np.mean(r2_rand) - np.mean(r2_spat),
            "RMSE_Inflation_Pct": ((np.mean(rmse_spat) - np.mean(rmse_rand)) / np.mean(rmse_rand)) * 100.0
        })

    cv_df = pd.DataFrame(cv_records)
    best_tree = HistGradientBoostingRegressor(
        max_iter=220, learning_rate=0.04, max_leaf_nodes=23,
        min_samples_leaf=15, l2_regularization=2.5, random_state=RANDOM_SEED
    )
    best_tree.fit(X, y)
    return cv_df, best_tree, X, y


def compute_shap_and_pdp(best_tree, X, y):
    print("[4/6] Computing TreeSHAP attributions and non-linear Partial Dependence grids...")
    method_used = "TreeSHAP"
    if HAS_SHAP:
        try:
            explainer = shap.TreeExplainer(best_tree)
            shap_vals = explainer.shap_values(X)
            mean_abs_attr = np.mean(np.abs(shap_vals), axis=0)
            signed_corr = [
                np.corrcoef(X.iloc[:, i].values, shap_vals[:, i])[0, 1]
                if np.std(X.iloc[:, i].values) > 1e-9 else 0.0
                for i in range(X.shape[1])
            ]
        except Exception:
            method_used = "Permutation_Importance"
            perm = permutation_importance(best_tree, X, y, n_repeats=10, random_state=RANDOM_SEED, n_jobs=-1)
            mean_abs_attr = perm.importances_mean
            signed_corr = [np.corrcoef(X.iloc[:, i].values, y)[0, 1] for i in range(X.shape[1])]
    else:
        method_used = "Permutation_Importance"
        perm = permutation_importance(best_tree, X, y, n_repeats=10, random_state=RANDOM_SEED, n_jobs=-1)
        mean_abs_attr = perm.importances_mean
        signed_corr = [np.corrcoef(X.iloc[:, i].values, y)[0, 1] for i in range(X.shape[1])]

    total_attr = np.sum(mean_abs_attr) + 1e-12
    clean_labels = [FEATURE_META.get(c, (c, c, "Submarket Controls"))[0] for c in X.columns]
    latex_labels = [FEATURE_META.get(c, (c, c, "Submarket Controls"))[1] for c in X.columns]
    domains = [FEATURE_META.get(c, (c, c, "Submarket Controls"))[2] for c in X.columns]

    shap_df = pd.DataFrame({
        "Feature": X.columns,
        "Clean_Label": clean_labels,
        "LaTeX_Label": latex_labels,
        "Domain": domains,
        "Mean_Abs_SHAP": mean_abs_attr,
        "Relative_Share_Pct": (mean_abs_attr / total_attr) * 100.0,
        "Direction_Rho": signed_corr,
        "Attribution_Method": method_used
    }).sort_values(by="Mean_Abs_SHAP", ascending=False).reset_index(drop=True)

    pdp_targets = [
        ("dist_to_historic_center_km", "Distance to Historic CBD (km)", "Polycentric & Enclave"),
        ("dist_industrial_subcenter_km", "Distance to Industrial Subcenter (km)", "Polycentric & Enclave"),
        ("ln_built_area", "Log Built Floor Area (ln m²)", "Structural Scale"),
        ("cv_prob_good_condition_mean", "CLIP Structural Conservation Score", "Computer Vision (CLIP)"),
        ("cv_modern_int_clean", "CLIP Modern Interior Finishes Score", "Computer Vision (CLIP)"),
        ("amenity_index_1km", "Log DENUE Amenity Density (1km)", "Neighborhood & DENUE")
    ]

    pdp_summaries = []
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.2), dpi=300)
    axes = axes.flatten()

    for idx, (col_name, clean_label, dom_name) in enumerate(pdp_targets):
        pdp_res = partial_dependence(best_tree, X, features=[col_name], grid_resolution=30, percentiles=(0.05, 0.95))
        grid_vals = pdp_res["grid_values"][0]
        avg_preds = pdp_res["average"][0]
        dom_color = DOMAIN_PALETTE.get(dom_name, "#1f4e78")

        if len(grid_vals) > 1:
            slopes = np.diff(avg_preds) / (np.diff(grid_vals) + 1e-9)
            max_steep_idx = int(np.argmax(np.abs(slopes)))
            inflection_x = float(grid_vals[max_steep_idx])
            inflection_y = float(avg_preds[max_steep_idx])
        else:
            inflection_x = float(grid_vals[0])
            inflection_y = float(avg_preds[0])

        pdp_range = float(np.max(avg_preds) - np.min(avg_preds))
        panel_letter = chr(65 + idx)

        axes[idx].plot(grid_vals, avg_preds, color=dom_color, linewidth=2.6)
        axes[idx].fill_between(grid_vals, np.min(avg_preds), avg_preds, color=dom_color, alpha=0.10)
        axes[idx].axhline(np.mean(avg_preds), color="#555555", linestyle=":", linewidth=1.0, alpha=0.8)
        axes[idx].axvline(inflection_x, color="#c0392b", linestyle="--", linewidth=1.3, alpha=0.85)
        axes[idx].scatter([inflection_x], [inflection_y], color="#c0392b", s=45, zorder=5)

        axes[idx].annotate(
            f"Threshold: {inflection_x:.2f}",
            xy=(inflection_x, inflection_y),
            xytext=(0.08, 0.86),
            textcoords="axes fraction",
            fontsize=8.5,
            fontweight="bold",
            color="#922b21",
            bbox=dict(boxstyle="round,pad=0.25", fc="#fdf2e9", ec="#c0392b", lw=0.8)
        )

        axes[idx].set_title(f"{panel_letter}. {clean_label}", fontsize=10.5, fontweight="bold", pad=8)
        axes[idx].set_xlabel(clean_label, fontsize=9.5)
        axes[idx].set_ylabel("Partial Dependence (Log Asking Price)", fontsize=9.5)
        axes[idx].grid(True, linestyle="--", alpha=0.35)

        pdp_summaries.append({
            "Feature": col_name,
            "Label": clean_label,
            "Domain": dom_name,
            "Grid_Min": float(grid_vals[0]),
            "Grid_Median": float(np.median(grid_vals)),
            "Grid_Max": float(grid_vals[-1]),
            "PDP_at_Min": float(avg_preds[0]),
            "PDP_at_Median": float(avg_preds[len(avg_preds) // 2]),
            "PDP_at_Max": float(avg_preds[-1]),
            "Total_Log_Price_Swing": pdp_range,
            "Pct_Price_Effect": (np.exp(pdp_range) - 1.0) * 100.0,
            "Primary_Threshold_X": inflection_x
        })

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig05_partial_dependence_curves_300dpi.png"), dpi=300, bbox_inches="tight")
    plt.close()

    return shap_df, pd.DataFrame(pdp_summaries)


# ==============================================================================
# 5. CARTOGRAPHY & ECONOMETRIC/ML FIGURES
# ==============================================================================
def load_or_create_cbd_thumbnail(metro_key):
    photo_filenames = {
        "Aguascalientes": "Plaza_de_la_Patria_Aguascalientes_aéreo.jpg",
        "Queretaro": "Vista_aérea_del_Jardin_Zenea,_Querétaro.jpg",
        "Zacatecas-Guadalupe": "Zacatecas_plaza_de_armas.jpg",
        "Leon": "Plaza_Mártires_Guanajuato.jpg"
    }
    fname = photo_filenames.get(metro_key, "")
    for d_cand in [INPUTS_DIR, CITY_PHOTOS_DIR, os.path.join(FALLBACK_BASE_DIR, "Inputs")]:
        cand = os.path.join(d_cand, fname)
        if fname and os.path.exists(cand):
            try:
                im = Image.open(cand).convert("RGB")
                im = ImageOps.fit(im, (260, 180), Image.Resampling.LANCZOS)
                im = ImageOps.expand(im, border=4, fill="white")
                im = ImageOps.expand(im, border=2, fill="#1f4e78")
                return np.asarray(im)
            except Exception:
                pass

    im = Image.new("RGB", (260, 180), color="#f4f6f9")
    draw = ImageDraw.Draw(im)
    draw.rectangle([4, 4, 255, 175], outline="#1f4e78", width=3)
    draw.rectangle([12, 12, 247, 46], fill="#1f4e78")
    title_str = METRO_DISPLAY_NAMES.get(metro_key, metro_key).split("(")[0].strip()
    landmark_str = METRO_DISPLAY_NAMES.get(metro_key, "(").split("(")[-1].replace(")", "").strip()
    draw.text((20, 22), f"Historic Core: {title_str}", fill="white")
    draw.text((20, 65), f"Landmark: {landmark_str}", fill="#1f4e78")
    draw.text((20, 92), f"Coords: {METRO_CENTERS[metro_key][0]:.4f} N,", fill="#333333")
    draw.text((20, 112), f"        {METRO_CENTERS[metro_key][1]:.4f} W", fill="#333333")
    return np.asarray(im)


def generate_all_publication_figures(df, gdf_polygons, subcenters_dict, ols_models, cv_df, shap_df):
    print("[5/6] Generating 300 DPI Cartography, Forest Plot, Gradients, TreeSHAP & CV Figures...")
    sns.set_theme(style="whitegrid")

    # --------------------------------------------------------------------------
    # FIGURE 1: 2x2 Metropolitan Study Area Map with Exogenous DENUE Subcenters
    # --------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(16, 14.5), dpi=300)
    axes = axes.flatten()
    metro_order = ["Queretaro", "Aguascalientes", "Leon", "Zacatecas-Guadalupe"]

    for idx, m in enumerate(metro_order):
        ax = axes[idx]
        panel_letter = chr(65 + idx)
        cx, cy = subcenters_dict[m]["cbd"]

        circle_15km = plt.Circle((cx, cy), 15000, color="#1f4e78", fill=True, facecolor="#ebf5fb", alpha=0.22, linestyle="--", linewidth=1.4, zorder=1)
        circle_15km_edge = plt.Circle((cx, cy), 15000, color="#1f4e78", fill=False, linestyle="--", linewidth=1.4, zorder=4)
        circle_7km_edge = plt.Circle((cx, cy), 7000, color="#7f8c8d", fill=False, linestyle=":", linewidth=1.0, zorder=3)
        ax.add_patch(circle_15km)
        ax.add_patch(circle_15km_edge)
        ax.add_patch(circle_7km_edge)

        if gdf_polygons is not None:
            sub_poly = gdf_polygons[gdf_polygons["metro_shp"] == m].copy()
            if len(sub_poly) > 0:
                poly_dists = sub_poly.geometry.centroid.distance(Point(cx, cy))
                sub_poly = sub_poly[poly_dists <= 16500].copy()
                open_polys = sub_poly[sub_poly["is_gated_Poly"] == 0]
                gated_polys = sub_poly[sub_poly["is_gated_Poly"] == 1]
                if len(open_polys) > 0:
                    open_polys.plot(ax=ax, facecolor="#d5dbdb", edgecolor="#85929e", linewidth=0.35, alpha=0.75, zorder=2)
                if len(gated_polys) > 0:
                    gated_polys.plot(ax=ax, facecolor="#aed6f1", edgecolor="#2874a6", linewidth=0.45, alpha=0.85, zorder=2)

        sub_df = df[df["metro_area"] == m]
        open_pts = sub_df[sub_df["is_planned_gated_enclave"] == 0]
        gated_pts = sub_df[sub_df["is_planned_gated_enclave"] == 1]

        ax.scatter(open_pts["x_metric"], open_pts["y_metric"], s=14, c="#7f8c8d", alpha=0.55, edgecolors="none", zorder=5)
        ax.scatter(gated_pts["x_metric"], gated_pts["y_metric"], s=16, c="#008080", alpha=0.65, edgecolors="none", zorder=5)

        ret_pts = subcenters_dict[m]["retail"]
        ind_pts = subcenters_dict[m]["industrial"]
        if len(ret_pts) > 0:
            ax.scatter(ret_pts[:, 0], ret_pts[:, 1], marker="^", s=115, c="#27ae60", edgecolors="black", linewidths=0.9, zorder=7)
        if len(ind_pts) > 0:
            ax.scatter(ind_pts[:, 0], ind_pts[:, 1], marker="s", s=110, c="#d35400", edgecolors="black", linewidths=0.9, zorder=7)

        ax.scatter([cx], [cy], marker="*", s=340, c="#c0392b", edgecolors="white", linewidths=1.1, zorder=9)

        thumb_arr = load_or_create_cbd_thumbnail(m)
        imagebox = OffsetImage(thumb_arr, zoom=0.52)
        ab = AnnotationBbox(
            imagebox, xy=(cx, cy), xybox=(cx - 12200, cy + 11800),
            xycoords="data", boxcoords="data", pad=0.0, frameon=False,
            arrowprops=dict(arrowstyle="->,head_width=0.35,head_length=0.35", connectionstyle="arc3,rad=-0.18", color="#c0392b", lw=1.8),
            zorder=10
        )
        ax.add_artist(ab)

        na_x, na_y = cx + 14200, cy + 12800
        ax.annotate(
            "N", xy=(na_x, na_y + 2400), xytext=(na_x, na_y),
            arrowprops=dict(facecolor="#1c2833", edgecolor="#1c2833", width=3.0, headwidth=9.0, headlength=9.0),
            ha="center", va="center", fontsize=11, fontweight="bold", color="#1c2833", zorder=10
        )

        sb_x0, sb_x1 = cx + 9500, cx + 14500
        sb_y = cy - 14800
        ax.plot([sb_x0, sb_x1], [sb_y, sb_y], color="#1c2833", linewidth=3.2, zorder=10)
        ax.plot([sb_x0, sb_x0], [sb_y - 450, sb_y + 450], color="#1c2833", linewidth=2.0, zorder=10)
        ax.plot([sb_x1, sb_x1], [sb_y - 450, sb_y + 450], color="#1c2833", linewidth=2.0, zorder=10)
        ax.text((sb_x0 + sb_x1) / 2.0, sb_y + 700, "5 km", ha="center", va="bottom", fontsize=9.5, fontweight="bold", color="#1c2833", zorder=10)

        ax.set_xlim(cx - 17800, cx + 17800)
        ax.set_ylim(cy - 17500, cy + 17800)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])

        gated_pct = sub_df["is_planned_gated_enclave"].mean() * 100.0 if len(sub_df) else 0.0
        ax.set_title(
            f"{panel_letter}. {METRO_DISPLAY_NAMES[m]}  [N = {len(sub_df):,}; {gated_pct:.1f}% Gated]",
            fontsize=11.5, fontweight="bold", pad=10, color="#1c2833"
        )

    legend_handles = [
        mlines.Line2D([], [], color="white", marker="*", markerfacecolor="#c0392b", markeredgecolor="black", markersize=15, label="Historic CBD Anchor"),
        mlines.Line2D([], [], color="white", marker="^", markerfacecolor="#27ae60", markeredgecolor="black", markersize=10, label="DENUE Retail Subcenter (Weighted K-Means)"),
        mlines.Line2D([], [], color="white", marker="s", markerfacecolor="#d35400", markeredgecolor="black", markersize=10, label="DENUE Industrial Subcenter (Weighted K-Means)"),
        mpatches.Patch(facecolor="#aed6f1", edgecolor="#2874a6", label="Planned Gated Enclave Polygon (CONAPO)"),
        mpatches.Patch(facecolor="#d5dbdb", edgecolor="#85929e", label="Open Colonia Polygon (CONAPO)"),
        mlines.Line2D([], [], color="white", marker="o", markerfacecolor="#008080", markersize=7, label="Gated Enclave Listing"),
        mlines.Line2D([], [], color="white", marker="o", markerfacecolor="#7f8c8d", markersize=7, label="Open Colonia Listing"),
        mlines.Line2D([], [], color="#1f4e78", linestyle="--", linewidth=1.4, label="15 km Core Boundary")
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4, fontsize=10, frameon=True, facecolor="white", edgecolor="#cccccc", bbox_to_anchor=(0.5, 0.01))
    plt.tight_layout(rect=[0.0, 0.06, 1.0, 1.0])
    plt.savefig(os.path.join(FIGURES_DIR, "fig01_metro_study_area_map_300dpi.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # --------------------------------------------------------------------------
    # FIGURE 2: Econometric Semi-Elasticities Forest Plot (Model 5, 95% CIs)
    # --------------------------------------------------------------------------
    m5 = ols_models["Model 5 (+Computer Vision)"]
    forest_vars = [
        "ln_built_area", "bathrooms", "bedrooms",
        "dist_to_historic_center_km", "dist_retail_subcenter_km", "dist_industrial_subcenter_km",
        "is_planned_gated_enclave", "is_planned_gated_enclave:dist_to_historic_center_km",
        "z_marginalization", "census_internet_share", "amenity_index_1km", "nightlife_index_1km",
        "cv_prob_good_condition_mean", "cv_modern_int_clean",
        "cv_share_interior", "cv_prob_greenery_max", "cv_has_cgi_render", "ln_sharpness"
    ]
    f_rows = []
    conf_int = m5.conf_int(alpha=0.05)
    for v in forest_vars:
        if v in m5.params:
            clean_lbl, _, dom = FEATURE_META.get(v, (v, v, "Submarket Controls"))
            f_rows.append({
                "var": v, "label": clean_lbl, "domain": dom,
                "coef": float(m5.params[v]), "ci_low": float(conf_int.loc[v, 0]),
                "ci_high": float(conf_int.loc[v, 1]), "pval": float(m5.pvalues[v])
            })
    f_df = pd.DataFrame(f_rows).iloc[::-1].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10.5, 7.6), dpi=300)
    ax.axvline(0.0, color="#333333", linestyle="--", linewidth=1.2, zorder=1)

    for idx, row in f_df.iterrows():
        c_hex = DOMAIN_PALETTE.get(row["domain"], "#1f4e78")
        ax.errorbar(
            row["coef"], idx,
            xerr=[[row["coef"] - row["ci_low"]], [row["ci_high"] - row["coef"]]],
            fmt="o", color=c_hex, ecolor=c_hex, elinewidth=2.2, capsize=4, markersize=7.5, zorder=3
        )
        stars = "***" if row["pval"] < 0.01 else ("**" if row["pval"] < 0.05 else ("*" if row["pval"] < 0.10 else ""))
        offset_x = 0.025 if row["coef"] >= 0 else -0.025
        align_h = "left" if row["coef"] >= 0 else "right"
        ax.text(row["coef"] + offset_x, idx + 0.18, f"{row['coef']:.3f}{stars}", va="center", ha=align_h, fontsize=8.5, fontweight="bold", color=c_hex)

    ax.set_yticks(range(len(f_df)))
    ax.set_yticklabels(f_df["label"], fontsize=9.8)
    ax.set_xlabel("Hedonic Semi-Elasticity / Marginal Coefficient (Log Asking Price, 95% Colonia-Clustered CI)", fontsize=10.5)
    ax.grid(True, axis="x", linestyle="--", alpha=0.4)

    dom_patches = [mpatches.Patch(color=DOMAIN_PALETTE[d], label=d) for d in ["Structural Scale", "Polycentric & Enclave", "Neighborhood & DENUE", "Computer Vision (CLIP)"]]
    ax.legend(handles=dom_patches, loc="lower right", frameon=True, fontsize=9.2, title="Covariate Domain", title_fontsize=9.5)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig02_hedonic_forest_plot_300dpi.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # --------------------------------------------------------------------------
    # FIGURE 3: Spatial Gradients by Settlement Morphology (2x2)
    # --------------------------------------------------------------------------
    df["Settlement_Type"] = np.where(df["is_planned_gated_enclave"] == 1, "Planned Gated Enclave", "Open Colonia")
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5), dpi=300, sharey=True)
    axes = axes.flatten()
    morph_colors = {"Planned Gated Enclave": "#008080", "Open Colonia": "#c0392b"}

    for idx, m in enumerate(metro_order):
        ax = axes[idx]
        sub = df[df["metro_area"] == m]
        panel_letter = chr(65 + idx)
        for s_type in ["Open Colonia", "Planned Gated Enclave"]:
            s_sub = sub[sub["Settlement_Type"] == s_type]
            if len(s_sub) > 3:
                sns.regplot(
                    data=s_sub, x="dist_to_historic_center_km", y="ln_price_m2",
                    ax=ax, color=morph_colors[s_type], label=s_type,
                    scatter_kws={"alpha": 0.28, "s": 20}, line_kws={"linewidth": 2.4}
                )
        ax.set_title(f"{panel_letter}. {METRO_DISPLAY_NAMES[m].split('(')[0].strip()}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Distance to Historic CBD (km)", fontsize=10)
        ax.set_ylabel("Log Asking Price per Square Meter (ln MXN/m²)" if idx % 2 == 0 else "", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.35)
        if idx == 0:
            ax.legend(loc="lower right", frameon=True, fontsize=9.5, title="Settlement Morphology")

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig03_spatial_gradients_by_morphology_300dpi.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # --------------------------------------------------------------------------
    # FIGURE 4: TreeSHAP Global Feature Importance
    # --------------------------------------------------------------------------
    top_shap = shap_df.head(15).iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10.5, 6.8), dpi=300)
    bar_colors = [DOMAIN_PALETTE.get(d, "#566573") for d in top_shap["Domain"]]

    bars = ax.barh(top_shap["Clean_Label"], top_shap["Relative_Share_Pct"], color=bar_colors, edgecolor="#1c2833", height=0.72, alpha=0.90)
    for bar, (_, r_row) in zip(bars, top_shap.iterrows()):
        w = bar.get_width()
        sign_str = "+" if r_row["Direction_Rho"] >= 0 else "-"
        ax.text(w + 0.5, bar.get_y() + bar.get_height() / 2.0, f"{w:.1f}% ({sign_str})", va="center", ha=align_h if False else "left", fontsize=8.8, fontweight="bold", color="#1c2833")

    ax.set_xlim(0, max(top_shap["Relative_Share_Pct"]) * 1.16)
    ax.set_xlabel("Relative Global Explanatory Importance Share (%)", fontsize=10.5)
    ax.grid(True, axis="x", linestyle="--", alpha=0.4)

    dom_handles = [mpatches.Patch(color=v, label=k) for k, v in DOMAIN_PALETTE.items()]
    ax.legend(handles=dom_handles, loc="lower right", frameon=True, fontsize=9.2, title="Feature Domain", title_fontsize=9.5)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig04_treeshap_feature_importance_300dpi.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # --------------------------------------------------------------------------
    # FIGURE 6: Random 5-Fold CV vs. Spatial-Block GroupKFold CV
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9.5, 5.4), dpi=300)
    x_pos = np.arange(len(cv_df))
    width = 0.36
    b1 = ax.bar(x_pos - width / 2, cv_df["Random_CV_R2"], width, yerr=cv_df["Random_CV_R2_SD"], capsize=4, label="Random 5-Fold Cross-Validation", color="#5499c7", edgecolor="#1c2833")
    b2 = ax.bar(x_pos + width / 2, cv_df["Spatial_CV_R2"], width, yerr=cv_df["Spatial_CV_R2_SD"], capsize=4, label="Spatial-Block 5-Fold Cross-Validation (GroupKFold)", color="#1f4e78", edgecolor="#1c2833")
    for b in list(b1) + list(b2):
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2.0, h + 0.02, f"{h:.3f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(cv_df["Model"], fontsize=10, fontweight="bold")
    ax.set_ylim(0, 0.95)
    ax.set_ylabel("Out-of-Sample R-Squared", fontsize=10.5)
    ax.legend(loc="upper left", frameon=True, fontsize=9.8)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig06_random_vs_spatial_cv_300dpi.png"), dpi=300, bbox_inches="tight")
    plt.close()


# ==============================================================================
# 6. EXPORT MAIN, APPENDIX & SUPPLEMENTAL LATEX TABLES TO Outputs_v8/Tables/
# ==============================================================================
def format_ols_table_df(models_dict, key_terms, include_aic=True):
    rows = []
    for raw_var, clean_label in key_terms:
        row_coef = {"Variable": clean_label}
        for m_name, mod in models_dict.items():
            if raw_var in mod.params:
                c_val = mod.params[raw_var]
                se_val = mod.bse[raw_var]
                p_val = mod.pvalues[raw_var]
                stars = "***" if p_val < 0.01 else ("**" if p_val < 0.05 else ("*" if p_val < 0.10 else ""))
                row_coef[m_name] = f"{c_val:.4f}{stars} ({se_val:.4f})"
            else:
                row_coef[m_name] = "---"
        rows.append(row_coef)

    stat_specs = [
        ("Observations ($N$)", lambda m: f"{int(m.nobs):,}"),
        ("R-Squared ($R^2$)", lambda m: f"{m.rsquared:.4f}"),
        ("Adjusted $R^2$", lambda m: f"{m.rsquared_adj:.4f}")
    ]
    if include_aic:
        stat_specs.append(("Akaike Info. Criterion (AIC)", lambda m: f"{m.aic:.1f}"))

    for stat_label, getter in stat_specs:
        r_stat = {"Variable": stat_label}
        for m_name, mod in models_dict.items():
            r_stat[m_name] = getter(mod)
        rows.append(r_stat)

    return pd.DataFrame(rows)


def write_clean_booktabs_tex(df_table, tex_path, caption, label, col_align=None, resize=True):
    if col_align is None:
        col_align = "l" + "r" * (df_table.shape[1] - 1)
    latex_body = df_table.to_latex(index=False, escape=False, column_format=col_align)
    lines = [r"\begin{table}[H]", r"\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}"]
    if resize:
        lines.append(r"\resizebox{\textwidth}{!}{%")
        lines.append(latex_body.strip())
        lines.append(r"}")
    else:
        lines.append(latex_body.strip())
    lines.append(r"\end{table}")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def export_all_tables(df, ols_models, city_models, robustness_models, cv_df, shap_df, pdp_df):
    print("[6/6] Exporting Main, Appendix, and Supplemental LaTeX/CSV tables to Outputs_v8/Tables/...")

    # 1. Table 1 (Main): Metropolitan Descriptive Statistics
    metro_desc = df.groupby("metro_area").agg(
        Listings_N=("listing_id", "count"),
        Colonias_N=("colonia_cvegeo", "nunique"),
        Price_Med_MXN=("price", "median"),
        Price_m2_Med=("price_m2", "median"),
        Built_m2_Med=("built_area_m2", "median"),
        Dist_CBD_Med=("dist_to_historic_center_km", "median"),
        Dist_Ret_Med=("dist_retail_subcenter_km", "median"),
        Dist_Ind_Med=("dist_industrial_subcenter_km", "median"),
        Amenity_Idx_Mean=("amenity_index_1km", "mean"),
        Gated_Share_Pct=("is_planned_gated_enclave", lambda s: round(s.mean() * 100, 1)),
        CGI_Share_Pct=("cv_has_cgi_render", lambda s: round(s.mean() * 100, 1)),
        Interior_Share_Pct=("cv_share_interior", lambda s: round(s.mean() * 100, 1)),
        Cond_Prob_Med=("cv_prob_good_condition_mean", "median"),
        Modern_Prob_Med=("cv_modern_int_clean", "median")
    ).reset_index().sort_values(by="Listings_N", ascending=False)

    metro_desc.to_csv(os.path.join(TABLES_DIR, "table01_metro_descriptive_stats.csv"), index=False)

    metro_tex_df = pd.DataFrame({
        "Metropolitan Area": [METRO_LATEX_NAMES.get(m, m) for m in metro_desc["metro_area"]],
        "Listings ($N$)": [f"{int(v):,}" for v in metro_desc["Listings_N"]],
        "Colonias ($C$)": [f"{int(v):,}" for v in metro_desc["Colonias_N"]],
        "Med. Price (MXN)": [f"{v:,.0f}" for v in metro_desc["Price_Med_MXN"]],
        "Med. Price/m$^2$": [f"{v:,.0f}" for v in metro_desc["Price_m2_Med"]],
        "Built m$^2$": [f"{v:.1f}" for v in metro_desc["Built_m2_Med"]],
        "CBD km": [f"{v:.2f}" for v in metro_desc["Dist_CBD_Med"]],
        "Ret. km": [f"{v:.2f}" for v in metro_desc["Dist_Ret_Med"]],
        "Ind. km": [f"{v:.2f}" for v in metro_desc["Dist_Ind_Med"]],
        "Amenity Idx": [f"{v:.2f}" for v in metro_desc["Amenity_Idx_Mean"]],
        "Gated (\\%)": [f"{v:.1f}" for v in metro_desc["Gated_Share_Pct"]],
        "CGI (\\%)": [f"{v:.1f}" for v in metro_desc["CGI_Share_Pct"]],
        "Conservation": [f"{v:.2f}" for v in metro_desc["Cond_Prob_Med"]],
        "Modern Fin.": [f"{v:.2f}" for v in metro_desc["Modern_Prob_Med"]]
    })
    write_clean_booktabs_tex(
        metro_tex_df,
        os.path.join(TABLES_DIR, "table01_metro_descriptive_stats.tex"),
        caption=r"Metropolitan Sample Distribution, DENUE Polycentric Gradients, and Computer Vision Medians ($\le 15\text{ km}$ Core)",
        label="tab:metro_desc"
    )

    # 2. Table 2 (Main): 5-Model Progressive Morphological & Visual Bundling OLS Sequence
    key_ols_terms = [
        ("Intercept", "Intercept"),
        ("dist_to_historic_center_km", FEATURE_META["dist_to_historic_center_km"][1]),
        ("dist_retail_subcenter_km", FEATURE_META["dist_retail_subcenter_km"][1]),
        ("dist_industrial_subcenter_km", FEATURE_META["dist_industrial_subcenter_km"][1]),
        ("dist_industrial_subcenter_km_sq", FEATURE_META["dist_industrial_subcenter_km_sq"][1]),
        ("is_planned_gated_enclave", FEATURE_META["is_planned_gated_enclave"][1]),
        ("is_planned_gated_enclave:dist_to_historic_center_km", FEATURE_META["is_planned_gated_enclave:dist_to_historic_center_km"][1]),
        ("z_marginalization", FEATURE_META["z_marginalization"][1]),
        ("census_internet_share", FEATURE_META["census_internet_share"][1]),
        ("amenity_index_1km", FEATURE_META["amenity_index_1km"][1]),
        ("nightlife_index_1km", FEATURE_META["nightlife_index_1km"][1]),
        ("ln_built_area", FEATURE_META["ln_built_area"][1]),
        ("bedrooms", FEATURE_META["bedrooms"][1]),
        ("bathrooms", FEATURE_META["bathrooms"][1]),
        ("cv_has_cgi_render", FEATURE_META["cv_has_cgi_render"][1]),
        ("cv_share_interior", FEATURE_META["cv_share_interior"][1]),
        ("cv_prob_good_condition_mean", FEATURE_META["cv_prob_good_condition_mean"][1]),
        ("cv_prob_greenery_max", FEATURE_META["cv_prob_greenery_max"][1]),
        ("cv_modern_int_clean", FEATURE_META["cv_modern_int_clean"][1]),
        ("ln_sharpness", FEATURE_META["ln_sharpness"][1])
    ]

    ols_prog_df = format_ols_table_df(ols_models, key_ols_terms, include_aic=True)
    ols_prog_df.to_csv(os.path.join(TABLES_DIR, "table02_hedonic_ols_progressive.csv"), index=False)
    write_clean_booktabs_tex(
        ols_prog_df,
        os.path.join(TABLES_DIR, "table02_hedonic_ols_progressive.tex"),
        caption=r"Progressive Colonia-Clustered Hedonic OLS Regressions Demonstrating Morphological and Visual Bundling ($\ln P_{icm}$)",
        label="tab:hedonic_ols",
        col_align="lccccc"
    )

    # 3. Table 3 (Main): Random vs. Spatial-Block CV Benchmarks
    cv_df.to_csv(os.path.join(TABLES_DIR, "table03_ml_spatial_cv_benchmarks.csv"), index=False)
    cv_tex_df = pd.DataFrame({
        "Estimator": cv_df["Model"],
        "Random CV $R^2$": [f"{v:.4f}" for v in cv_df["Random_CV_R2"]],
        "Random SD": [f"{v:.4f}" for v in cv_df["Random_CV_R2_SD"]],
        "Random RMSE": [f"{v:.4f}" for v in cv_df["Random_CV_RMSE"]],
        "Random MAE": [f"{v:.4f}" for v in cv_df["Random_CV_MAE"]],
        "Spatial CV $R^2$": [f"{v:.4f}" for v in cv_df["Spatial_CV_R2"]],
        "Spatial SD": [f"{v:.4f}" for v in cv_df["Spatial_CV_R2_SD"]],
        "Spatial RMSE": [f"{v:.4f}" for v in cv_df["Spatial_CV_RMSE"]],
        "Spatial MAE": [f"{v:.4f}" for v in cv_df["Spatial_CV_MAE"]],
        "Gap ($\\Delta R^2$)": [f"{v:.4f}" for v in cv_df["Generalization_Gap_R2"]],
        "RMSE Infl. (\\%)": [f"{v:.2f}\\%" for v in cv_df["RMSE_Inflation_Pct"]]
    })
    write_clean_booktabs_tex(
        cv_tex_df,
        os.path.join(TABLES_DIR, "table03_ml_spatial_cv_benchmarks.tex"),
        caption="Out-of-Sample Predictive Performance Under Random Five-Fold and Spatial-Block GroupKFold Cross-Validation",
        label="tab:ml_cv"
    )

    # 4. Table 4 (Main): TreeSHAP Global Attributions
    shap_df.to_csv(os.path.join(TABLES_DIR, "table04_treeshap_attributions.csv"), index=False)
    shap_top = shap_df.head(16).copy()
    shap_tex_df = pd.DataFrame({
        "Covariate": shap_top["LaTeX_Label"],
        "Thematic Domain": [d.replace("&", r"\&") for d in shap_top["Domain"]],
        "Mean $|\\phi_j|$": [f"{v:.4f}" for v in shap_top["Mean_Abs_SHAP"]],
        "Relative Share (\\%)": [f"{v:.2f}\\%" for v in shap_top["Relative_Share_Pct"]],
        "Direction ($\\rho$)": [f"{v:+.4f}" for v in shap_top["Direction_Rho"]]
    })
    write_clean_booktabs_tex(
        shap_tex_df,
        os.path.join(TABLES_DIR, "table04_treeshap_attributions.tex"),
        caption="Global Feature Attribution Shares and Directional Correlations in Gradient Boosted Decision Trees",
        label="tab:shap_imp",
        col_align="llrrr",
        resize=False
    )

    # 5. Table 5 (Main): Non-Linear Partial Dependence Inflection Thresholds
    pdp_df.to_csv(os.path.join(TABLES_DIR, "table05_pdp_nonlinear_thresholds.csv"), index=False)
    pdp_tex_df = pd.DataFrame({
        "Continuous Predictor": [FEATURE_META.get(f, (l, l, ""))[1] for f, l in zip(pdp_df["Feature"], pdp_df["Label"])],
        "Domain": [d.replace("&", r"\&") for d in pdp_df["Domain"]],
        "Grid Min": [f"{v:.3f}" for v in pdp_df["Grid_Min"]],
        "Grid Med": [f"{v:.3f}" for v in pdp_df["Grid_Median"]],
        "Grid Max": [f"{v:.3f}" for v in pdp_df["Grid_Max"]],
        "PDP Min": [f"{v:+.3f}" for v in pdp_df["PDP_at_Min"]],
        "PDP Max": [f"{v:+.3f}" for v in pdp_df["PDP_at_Max"]],
        "Log Swing": [f"{v:.3f}" for v in pdp_df["Total_Log_Price_Swing"]],
        "Price Effect (\\%)": [f"{v:.1f}\\%" for v in pdp_df["Pct_Price_Effect"]],
        "Threshold ($x_S^*$)": [f"{v:.3f}" for v in pdp_df["Primary_Threshold_X"]]
    })
    write_clean_booktabs_tex(
        pdp_tex_df,
        os.path.join(TABLES_DIR, "table05_pdp_nonlinear_thresholds.tex"),
        caption="Non-Linear Partial Dependence Swings and Primary Inflection Thresholds ($x_S^*$)",
        label="tab:pdp_thresholds",
        col_align="llrrrrrrrr"
    )

    # 6. Appendix Table A1: Robustness Checks (Including Within-Colonia Fixed Effects!)
    rob_df = format_ols_table_df(robustness_models, key_ols_terms, include_aic=True)
    rob_df.to_csv(os.path.join(TABLES_DIR, "tableA1_robustness_checks.csv"), index=False)
    rob_tex_df = rob_df.rename(columns={"R3: Core <= 10 km": r"R3: Core $\le 10\text{ km}$"})
    write_clean_booktabs_tex(
        rob_tex_df,
        os.path.join(TABLES_DIR, "tableA1_robustness_checks.tex"),
        caption="Robustness Checks Across Exact Colonia Matches, Real Camera Photos, Inner Core Radius, and Within-Colonia Fixed Effects",
        label="tab:robustness_a1",
        col_align="lcccc"
    )

    # 7. Appendix Table A2: Metropolitan Submarket Regressions
    city_terms = [
        t for t in key_ols_terms
        if ":" not in t[0] and t[0] not in ("dist_industrial_subcenter_km_sq", "census_internet_share", "ln_sharpness")
    ]
    city_models_latex = {METRO_LATEX_NAMES.get(k, k): v for k, v in city_models.items()}
    city_df = format_ols_table_df(city_models_latex, city_terms, include_aic=False)
    city_df.to_csv(os.path.join(TABLES_DIR, "tableA2_city_subsample_models.csv"), index=False)
    write_clean_booktabs_tex(
        city_df,
        os.path.join(TABLES_DIR, "tableA2_city_subsample_models.tex"),
        caption="Metropolitan Submarket Hedonic Regressions of Log Asking Price",
        label="tab:city_ols",
        col_align="lcccc"
    )

    # 8. Supplemental Table S1: Concentric Ring & Morphology Audit (True Observed Medians)
    df["dist_ring"] = pd.cut(
        df["dist_to_historic_center_km"],
        bins=[0, 3, 7, 15],
        labels=["Core (0-3 km)", "Intermediate (3-7 km)", "Periphery (7-15 km)"]
    )
    ring_audit = df.groupby(["dist_ring", "Settlement_Type"], observed=False).agg(
        N=("listing_id", "count"),
        Price_m2_Med=("price_m2", "median"),
        Price_m2_IQR=("price_m2", lambda s: s.quantile(0.75) - s.quantile(0.25)),
        Built_m2_Med=("built_area_m2", "median"),
        Amenity_Idx_Med=("amenity_index_1km", "median"),
        Cond_Mean=("cv_prob_good_condition_mean", "mean"),
        Modern_Mean=("cv_modern_int_clean", "mean"),
        CGI_Pct=("cv_has_cgi_render", lambda s: round(s.mean() * 100, 1))
    ).reset_index()
    ring_audit.to_csv(os.path.join(TABLES_DIR, "supp01_ring_morphology_audit.csv"), index=False)
    write_clean_booktabs_tex(
        ring_audit,
        os.path.join(TABLES_DIR, "supp01_ring_morphology_audit.tex"),
        caption="Concentric Distance Ring Audit of Unit Prices, Amenity Density, and Visual Quality by Settlement Morphology",
        label="tab:supp_rings"
    )

    # 9. Supplemental Table S2: Complete Feature Dictionary & Domain Mapping
    dict_rows = []
    for raw_k, (clean_l, tex_l, dom) in FEATURE_META.items():
        if raw_k in df.columns:
            s_num = pd.to_numeric(df[raw_k], errors="coerce")
            dict_rows.append({
                "Code_Column": raw_k.replace("_", r"\_"),
                "Publication_Label": tex_l,
                "Domain": dom.replace("&", r"\&"),
                "Mean": f"{s_num.mean():.3f}",
                "SD": f"{s_num.std():.3f}",
                "Median": f"{s_num.median():.3f}"
            })
    dict_df = pd.DataFrame(dict_rows)
    dict_df.to_csv(os.path.join(TABLES_DIR, "supp02_feature_dictionary_and_domains.csv"), index=False)
    write_clean_booktabs_tex(
        dict_df,
        os.path.join(TABLES_DIR, "supp02_feature_dictionary_and_domains.tex"),
        caption="Multimodal Variable Dictionary, Mathematical Symbols, Thematic Domains, and Sample Moments",
        label="tab:supp_dict"
    )

    # Print Console Summary of the 5-Model Bundling Sequence & CV Benchmarks
    print("\n" + "=" * 88)
    print("[SUCCESS] PIPELINE v8.0 COMPLETE! SUMMARY OF 5-MODEL PROGRESSIVE OLS SEQUENCE:")
    print("=" * 88)
    for m_name, mod in ols_models.items():
        g_c = mod.params.get("is_planned_gated_enclave", np.nan)
        g_p = mod.pvalues.get("is_planned_gated_enclave", np.nan)
        gi_c = mod.params.get("is_planned_gated_enclave:dist_to_historic_center_km", np.nan)
        gi_p = mod.pvalues.get("is_planned_gated_enclave:dist_to_historic_center_km", np.nan)
        print(f"  * {m_name:<28} | R2 = {mod.rsquared:.4f} | AIC = {mod.aic:.1f} | Gated = {g_c:+.4f} (p={g_p:.3f}) | Gated*CBD = {gi_c:+.4f} (p={gi_p:.3f})")

    print("\n--- RANDOM vs. SPATIAL-BLOCK CV BENCHMARKS ---")
    print(cv_df[["Model", "Random_CV_R2", "Spatial_CV_R2", "Generalization_Gap_R2", "RMSE_Inflation_Pct"]].to_string(index=False))
    print("=" * 88)
    print(f"All figures and LaTeX tables written to: {OUTPUTS_DIR}")


if __name__ == "__main__":
    df_master, gdf_polys, subcenters_map = load_v3_dataset_and_subcenters()
    ols_mods, city_mods, rob_mods = run_hedonic_ols_models(df_master)
    cv_bench_df, fitted_tree, X_mat, y_vec = evaluate_models_cv(df_master)
    shap_summary_df, pdp_summary_df = compute_shap_and_pdp(fitted_tree, X_mat, y_vec)
    generate_all_publication_figures(df_master, gdf_polys, subcenters_map, ols_mods, cv_bench_df, shap_summary_df)
    export_all_tables(df_master, ols_mods, city_mods, rob_mods, cv_bench_df, shap_summary_df, pdp_summary_df)
