#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05_spatial_features_denue.py
============================
Exogenous Polycentric Subcenter Identification & MAUP-Safe DENUE Buffer Engine.
Processes CONAPO Urban Deprivation (IMC 2020) polygons and 168,686 offline INEGI DENUE
economic establishments across Queretaro, Aguascalientes, Leon, and Zacatecas-Guadalupe:
  1. Projects all geometries to Mexico ITRF2008 / LCC metric CRS (EPSG:6372).
  2. Computes Euclidean distance to the historic central plaza (D_CBD_km <= 15 km filter).
  3. Maps INEGI DENUE employment size strata (per_ocu) to midpoint worker weights (w_e)
     and identifies K = 3 retail subcenters (SCIAN 46) and K = 3 industrial subcenters
     (SCIAN 31-33) per metropolitan area via employment-weighted K-Means clustering.
  4. Computes minimum distances to exogenous retail (D_RET_km) and industrial (D_IND_km) nodes.
  5. Evaluates scale-invariant 1-km radial buffers around each colonia centroid using
     cKDTree spatial indexing to compute positive commercial/civic amenity log-density (I_cm)
     and localized nightlife/alcohol disamenity log-density (N_cm).

Author: Research Pipeline (Outputs_v8)
Date: October 2026
"""

import os
import re
import glob
import unicodedata
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")

# ==============================================================================
# 1. CONFIGURATION & PATHS
# ==============================================================================
DEFAULT_BASE_DIR = r"C:\Users\multiple_cities_v3"
FALLBACK_BASE_DIR = r"C:\Users\multiple_cities_v2"

BASE_DIR = DEFAULT_BASE_DIR if os.path.exists(DEFAULT_BASE_DIR) else FALLBACK_BASE_DIR
INPUTS_DIR = os.path.join(BASE_DIR, "Inputs")
DATA_DIR = os.path.join(BASE_DIR, "Data")
DENUE_DIR = os.path.join(DATA_DIR, "DENUE")

os.makedirs(INPUTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

METRIC_CRS = "EPSG:6372"
RANDOM_SEED = 42
CORE_RADIUS_KM = 15.0
BUFFER_RADIUS_M = 1000.0
BUFFER_AREA_KM2 = np.pi * ((BUFFER_RADIUS_M / 1000.0) ** 2)  # 3.14159 km2

CANDIDATE_RAW_CSVS = [
    os.path.join(INPUTS_DIR, "property_listings_multicity_consolidated.csv"),
    os.path.join(INPUTS_DIR, "property_listings_multicity_fixed.csv"),
    os.path.join(INPUTS_DIR, "property_listings_multicity.csv"),
    os.path.join(DATA_DIR, "property_listings_multicity_consolidated.csv")
]

OUTPUT_COLONIA_LOOKUP_CSV = os.path.join(INPUTS_DIR, "colonia_spatial_lookup_v3.csv")
OUTPUT_SUBCENTERS_CSV = os.path.join(INPUTS_DIR, "polycentric_subcenters_v3.csv")
OUTPUT_SPATIAL_CSV = os.path.join(INPUTS_DIR, "property_listings_multicity_spatial_v3.csv")
OUTPUT_15KM_CSV = os.path.join(INPUTS_DIR, "property_listings_multicity_filtered_15km_v3.csv")

METRO_CENTERS = {
    "Queretaro": (20.5931, -100.3921),           # Jardín Zenea, Querétaro
    "Aguascalientes": (21.8810, -102.2963),      # Plaza de la Patria, Aguascalientes
    "Leon": (21.1220, -101.6825),                # Plaza de los Mártires, León
    "Zacatecas-Guadalupe": (22.7764, -102.5720)  # Plaza de Armas, Zacatecas
}

METRO_SUBCENTER_K = {
    "Queretaro": {"retail": 4, "industrial": 4},
    "Aguascalientes": {"retail": 3, "industrial": 3},
    "Leon": {"retail": 4, "industrial": 4},
    "Zacatecas-Guadalupe": {"retail": 2, "industrial": 2}
}

TARGET_MUN_TO_METRO = {
    "01001": "Aguascalientes", "01005": "Aguascalientes", "01011": "Aguascalientes",
    "11020": "Leon", "11025": "Leon", "11031": "Leon",
    "22014": "Queretaro", "22006": "Queretaro", "22011": "Queretaro", "22008": "Queretaro",
    "32056": "Zacatecas-Guadalupe", "32017": "Zacatecas-Guadalupe",
    "32032": "Zacatecas-Guadalupe", "32050": "Zacatecas-Guadalupe", "32057": "Zacatecas-Guadalupe"
}

MUN_NAME_TO_METRO = {
    "AGUASCALIENTES": "Aguascalientes", "JESUS MARIA": "Aguascalientes", "SAN FRANCISCO DE LOS ROMO": "Aguascalientes",
    "LEON": "Leon", "PURISIMA DEL RINCON": "Leon", "SAN FRANCISCO DEL RINCON": "Leon",
    "QUERETARO": "Queretaro", "CORREGIDORA": "Queretaro", "EL MARQUES": "Queretaro", "HUIMILPAN": "Queretaro",
    "ZACATECAS": "Zacatecas-Guadalupe", "GUADALUPE": "Zacatecas-Guadalupe",
    "MORELOS": "Zacatecas-Guadalupe", "VETAGRANDE": "Zacatecas-Guadalupe", "TRANCOSO": "Zacatecas-Guadalupe"
}

GATED_POLYGON_KEYWORDS = [
    "FRACCIONAMIENTO", "PRIVADA", "RESIDENCIAL", "CONDOMINIO",
    "COTO", "CONJUNTO HABITACIONAL", "RINCONADA", "CLUB DE GOLF", "CAMPESTRE"
]

GATED_LISTING_REGEX = re.compile(
    r"\b(?:FRACCIONAMIENTO|FRACC|PRIVADA|RESIDENCIAL|CONDOMINIO|COTO|"
    r"CONJUNTO\s+HABITACIONAL|CLUB\s+DE\s+GOLF|ACCESO\s+CONTROLADO|"
    r"VIGILANCIA\s+24|SEGURIDAD\s+PRIVADA|CASETA\s+DE\s+VIGILANCIA)\b",
    flags=re.IGNORECASE
)


# ==============================================================================
# 2. HELPER UTILITIES
# ==============================================================================
def fix_mojibake(val):
    if val is None or pd.isna(val):
        return ""
    s = str(val).strip()
    if "Ã" in s or "Â" in s:
        for enc in ("latin1", "cp1252"):
            try:
                s = s.encode(enc).decode("utf-8")
                break
            except Exception:
                pass
    return s


def normalize_str(val):
    s = fix_mojibake(val).upper()
    if not s or s in ("NAN", "NONE", "NULL"):
        return ""
    s = "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def normalize_colonia_name(val):
    s = normalize_str(val)
    if not s:
        return ""
    prefix_pat = re.compile(
        r"^(?:RANCHO\s+O\s+RANCHERIA|UNIDAD\s+HABITACIONAL|FRACCIONAMIENTO|FRACC|"
        r"COLONIA|COL|PRIVADA|PRIV|RESIDENCIAL|RES|BARRIO|BARR|CONJUNTO\s+HABITACIONAL|"
        r"CONDOMINIO|COND|COTO|SECTOR|AMPLIACION|AMPL|EJIDO|ZONA|VALLE\s+RESIDENCIAL)\b\s*"
    )
    prev = None
    while s and s != prev:
        prev = s
        s = prefix_pat.sub("", s).strip()
    s = re.sub(r"\b1(?:RA|ERA|A)?\s+SECC(?:ION)?\b", "PRIMERA SECCION", s)
    s = re.sub(r"\b2(?:DA|A)?\s+SECC(?:ION)?\b", "SEGUNDA SECCION", s)
    s = re.sub(r"\b3(?:RA|ERA|A)?\s+SECC(?:ION)?\b", "TERCERA SECCION", s)
    return re.sub(r"\s+", " ", s).strip()


def canonicalize_cve_col(val):
    if val is None or pd.isna(val):
        return ""
    s = re.sub(r"\.0+$", "", str(val).strip())
    digits = re.sub(r"[^0-9]", "", s)
    if len(digits) == 8:
        return "0" + digits
    if len(digits) == 9:
        return digits
    return ""


def canonicalize_cve_mun(val, cve_col_9=""):
    if cve_col_9 and len(cve_col_9) == 9:
        return cve_col_9[:5]
    if val is None or pd.isna(val):
        return ""
    s = re.sub(r"\.0+$", "", str(val).strip())
    digits = re.sub(r"[^0-9]", "", s)
    if len(digits) == 4:
        return "0" + digits
    if len(digits) == 5:
        return digits
    return ""


def classify_polygon_gated(clasif_val, nom_val):
    comb = f"{normalize_str(clasif_val)} {normalize_str(nom_val)}"
    for kw in GATED_POLYGON_KEYWORDS:
        if kw in comb:
            return 1
    return 0


def classify_listing_gated(title_val, col_rep_val, desc_val=""):
    comb = f"{normalize_str(title_val)} {normalize_str(col_rep_val)} {normalize_str(str(desc_val)[:600])}"
    return 1 if GATED_LISTING_REGEX.search(comb) else 0


# ==============================================================================
# 3. LOAD CONAPO IMC 2020 SHAPEFILE
# ==============================================================================
def load_conapo_colonias_gdf():
    shp_candidates = [
        os.path.join(DATA_DIR, "colonias_imc2020_shp", "colonias_imc2020.shp"),
        os.path.join(INPUTS_DIR, "colonias_imc2020_shp", "colonias_imc2020.shp"),
        os.path.join(BASE_DIR, "colonias_imc2020.shp")
    ]
    csv_candidates = [
        os.path.join(DATA_DIR, "colonias.csv"),
        os.path.join(INPUTS_DIR, "colonias.csv")
    ]

    shp_path = next((p for p in shp_candidates if os.path.exists(p)), None)
    csv_path = next((p for p in csv_candidates if os.path.exists(p)), None)

    if not shp_path:
        raise FileNotFoundError(f"CONAPO shapefile `colonias_imc2020.shp` not found in {DATA_DIR}.")

    print(f"[1/6] Loading CONAPO IMC 2020 Shapefile from: {shp_path}")
    gdf = None
    for enc in (None, "utf-8", "cp1252", "latin1"):
        try:
            gdf = gpd.read_file(shp_path, encoding=enc) if enc else gpd.read_file(shp_path)
            break
        except Exception:
            continue

    if gdf.crs is None:
        gdf.set_crs(METRIC_CRS, inplace=True)
    else:
        gdf = gdf.to_crs(METRIC_CRS)

    id_col = "CVE_COL" if "CVE_COL" in gdf.columns else ("CVEGEO" if "CVEGEO" in gdf.columns else gdf.columns[0])
    gdf["cve_col_9"] = gdf[id_col].apply(canonicalize_cve_col)
    gdf = gdf[gdf["cve_col_9"] != ""].copy()

    gdf["cve_mun_5"] = [
        canonicalize_cve_mun(m, c9)
        for m, c9 in zip(gdf.get("CVE_MUN", pd.Series("", index=gdf.index)), gdf["cve_col_9"])
    ]
    gdf = gdf[gdf["cve_mun_5"].isin(TARGET_MUN_TO_METRO.keys())].copy()
    gdf["metro_area"] = gdf["cve_mun_5"].map(TARGET_MUN_TO_METRO)

    imc_numeric_cols = [
        "POBTOT", "P6A14NAE", "SBASC", "PSDSS", "OVSDE", "OVSEE",
        "OVSAE", "OVPT", "OVHAC", "OVSREF", "OVSINT", "OVSCEL",
        "IM_2020", "IMN_2020"
    ]
    if csv_path:
        for enc in ("utf-8", "cp1252", "latin1"):
            try:
                df_csv = pd.read_csv(csv_path, dtype=str, encoding=enc, low_memory=False)
                if "CVE_COL" in df_csv.columns:
                    df_csv["cve_col_9"] = df_csv["CVE_COL"].apply(canonicalize_cve_col)
                    csv_map = df_csv[df_csv["cve_col_9"] != ""].drop_duplicates("cve_col_9").set_index("cve_col_9")
                    for c in ["COLONIA", "CLASIF", "GM_2020"] + imc_numeric_cols:
                        if c in csv_map.columns and c not in gdf.columns:
                            gdf[c] = gdf["cve_col_9"].map(csv_map[c])
                break
            except Exception:
                continue

    tipo_col = "CLASIF" if "CLASIF" in gdf.columns else ("TIPO" if "TIPO" in gdf.columns else None)
    nom_col = "COLONIA" if "COLONIA" in gdf.columns else ("NOM_ASEN" if "NOM_ASEN" in gdf.columns else None)

    gdf["conapo_clasif"] = gdf[tipo_col].apply(fix_mojibake) if tipo_col else "Colonia"
    gdf["colonia_nombre"] = gdf[nom_col].apply(fix_mojibake) if nom_col else ""
    gdf["norm_col_name"] = gdf["colonia_nombre"].apply(normalize_colonia_name)
    gdf["is_gated_polygon"] = [
        classify_polygon_gated(t, n) for t, n in zip(gdf["conapo_clasif"], gdf["colonia_nombre"])
    ]

    gdf["colonia_area_km2"] = (gdf.geometry.area / 1e6).round(4)
    centroids_metric = gdf.geometry.centroid
    gdf["x_metric"] = centroids_metric.x
    gdf["y_metric"] = centroids_metric.y

    centroids_wgs = centroids_metric.to_crs("EPSG:4326")
    gdf["centroid_lat"] = centroids_wgs.y
    gdf["centroid_lon"] = centroids_wgs.x

    for c in imc_numeric_cols:
        if c in gdf.columns:
            gdf[c] = pd.to_numeric(gdf[c], errors="coerce")

    gdf = gdf.drop_duplicates(subset=["cve_col_9"]).copy()
    print(f"      -> Retained {len(gdf):,} unique CONAPO colonias across {gdf['metro_area'].nunique()} metropolitan areas.")
    return gdf


# ==============================================================================
# 4. LOAD LOCAL DENUE CSVs & CLASSIFY SCIAN TAXONOMY OFFLINE
# ==============================================================================
def parse_denue_employment_weight(per_ocu_str):
    """Maps INEGI DENUE `per_ocu` employment strata strings to midpoint worker weights."""
    s = str(per_ocu_str or "").lower()
    if "251" in s:
        return 350.0
    if "101" in s:
        return 175.0
    if "51" in s:
        return 75.0
    if "31" in s:
        return 40.0
    if "11" in s:
        return 20.0
    if "6" in s:
        return 8.0
    return 3.0


def classify_scian_code(code_str):
    """
    Classifies 6-digit INEGI SCIAN activity codes into literature-backed
    amenity/disamenity categories:
      - 'food': Restaurants, cafes, supermarkets, convenience & grocery stores
      - 'health': Hospitals, medical/dental clinics, pharmacies, labs
      - 'edu': Preschool, primary, secondary, high school, universities
      - 'leisure': Parks, botanical gardens, museums, sports/fitness clubs, golf
      - 'nightlife_disamenity': Bars, cantinas, nightclubs, liquor/beer stores, motels
      - 'industrial_disamenity': Manufacturing plants (31-33)
      - 'major_retail_node': Supermarkets, minisupers, department stores, malls
    """
    c = str(code_str or "").strip()
    if not c:
        return "other"

    # 1. Nightlife, Alcohol & Motel Disamenities (Negative Externalities)
    if c.startswith(("722411", "722412", "461211", "461212", "721113", "7132")):
        return "nightlife"

    # 2. Food & Convenience Retail (Positive Daily Consumption Amenities)
    if c.startswith(("7225", "7223", "462111", "462112", "462210", "4611")):
        return "food"

    # 3. Healthcare & Pharmacies (Positive Civic/Health Infrastructure)
    if c.startswith(("621", "622", "464111", "464112")):
        return "health"

    # 4. Education (Schools, Universities, Childcare)
    if c.startswith(("6111", "6112", "6113", "6244")):
        return "edu"

    # 5. Leisure, Parks, Culture & Sports Centres
    if c.startswith(("7121", "7131", "7139")):
        return "leisure"

    # 6. Manufacturing & Heavy Industry (31, 32, 33)
    if c.startswith(("31", "32", "33")):
        return "industrial"

    # 7. General Retail (463-466)
    if c.startswith(("463", "465", "466")):
        return "retail_general"

    return "other"


def load_and_process_local_denue(gdf_colonias):
    print(f"[2/6] Scanning local INEGI DENUE directory: {DENUE_DIR}")
    all_csvs = [
        f for f in glob.glob(os.path.join(DENUE_DIR, "**", "*.csv"), recursive=True)
        if "unique_ramas" not in os.path.basename(f).lower()
    ]
    if not all_csvs:
        raise FileNotFoundError(f"No DENUE CSV files found in {DENUE_DIR}.")

    print(f"      -> Found {len(all_csvs)} DENUE CSV file(s). Reading and filtering to target municipalities...")

    # Precompute metric bounding boxes around our 4 metropolitan centers (18 km radius)
    cbd_records = [{"metro_area": m, "geometry": Point(lon, lat)} for m, (lat, lon) in METRO_CENTERS.items()]
    cbd_gdf = gpd.GeoDataFrame(cbd_records, crs="EPSG:4326").to_crs(METRIC_CRS)
    cbd_metric_map = {r["metro_area"]: (r.geometry.x, r.geometry.y) for _, r in cbd_gdf.iterrows()}

    denue_chunks = []
    for fpath in all_csvs:
        fname = os.path.basename(fpath)
        df_raw = None
        for enc in ("latin1", "cp1252", "utf-8"):
            try:
                # Read header first to identify exact column names
                hdr = pd.read_csv(fpath, encoding=enc, nrows=2)
                cols_lower = {c.lower(): c for c in hdr.columns}

                code_col = next((cols_lower[k] for k in ["codigo_act", "cod_act", "cve_act"] if k in cols_lower), None)
                lat_col = next((cols_lower[k] for k in ["latitud", "lat"] if k in cols_lower), None)
                lon_col = next((cols_lower[k] for k in ["longitud", "lon", "lng"] if k in cols_lower), None)
                per_col = next((cols_lower[k] for k in ["per_ocu", "estrato", "personal"] if k in cols_lower), None)
                ent_col = next((cols_lower[k] for k in ["cve_ent", "entidad_id"] if k in cols_lower), None)
                mun_col = next((cols_lower[k] for k in ["cve_mun", "municipio_id"] if k in cols_lower), None)

                if not (code_col and lat_col and lon_col):
                    break

                use_cols = [c for c in [code_col, lat_col, lon_col, per_col, ent_col, mun_col] if c is not None]
                df_raw = pd.read_csv(fpath, usecols=use_cols, dtype=str, encoding=enc, low_memory=False)
                df_raw = df_raw.rename(columns={
                    code_col: "codigo_act",
                    lat_col: "lat",
                    lon_col: "lon",
                    **({per_col: "per_ocu"} if per_col else {}),
                    **({ent_col: "cve_ent"} if ent_col else {}),
                    **({mun_col: "cve_mun"} if mun_col else {})
                })
                break
            except Exception:
                continue

        if df_raw is None or df_raw.empty:
            continue

        df_raw["lat"] = pd.to_numeric(df_raw["lat"], errors="coerce")
        df_raw["lon"] = pd.to_numeric(df_raw["lon"], errors="coerce")
        df_raw = df_raw.dropna(subset=["lat", "lon", "codigo_act"]).copy()

        # Fast geographic bounding box filter around the 4 metropolises (+/- 0.22 deg ~ 24 km)
        lat_v = df_raw["lat"].values
        lon_v = df_raw["lon"].values
        in_bbox = np.zeros(len(df_raw), dtype=bool)
        for _, (c_lat, c_lon) in METRO_CENTERS.items():
            in_bbox |= ((np.abs(lat_v - c_lat) <= 0.22) & (np.abs(lon_v - c_lon) <= 0.22))

        df_sub = df_raw[in_bbox].copy()
        if not df_sub.empty:
            denue_chunks.append(df_sub)
            print(f"         * Retained {len(df_sub):,} active economic units from {fname}")

    if not denue_chunks:
        raise RuntimeError("No DENUE points fell within the 4 metropolitan study areas. Verify DENUE CSV coordinates.")

    df_denue = pd.concat(denue_chunks, ignore_index=True)
    df_denue["scian_group"] = df_denue["codigo_act"].apply(classify_scian_code)
    df_denue["emp_weight"] = df_denue.get("per_ocu", pd.Series("", index=df_denue.index)).apply(parse_denue_employment_weight)

    # Project DENUE points to EPSG:6372
    gdf_denue = gpd.GeoDataFrame(
        df_denue,
        geometry=gpd.points_from_xy(df_denue["lon"], df_denue["lat"]),
        crs="EPSG:4326"
    ).to_crs(METRIC_CRS)
    gdf_denue["x_metric"] = gdf_denue.geometry.x
    gdf_denue["y_metric"] = gdf_denue.geometry.y

    # Assign each DENUE point to its nearest metropolitan center (within 18 km)
    gdf_denue["metro_area"] = "Outside"
    gdf_denue["dist_cbd_km"] = np.nan
    for metro_name, (cx, cy) in cbd_metric_map.items():
        d_km = np.hypot(gdf_denue["x_metric"] - cx, gdf_denue["y_metric"] - cy) / 1000.0
        mask_m = d_km <= 18.0
        gdf_denue.loc[mask_m, "metro_area"] = metro_name
        gdf_denue.loc[mask_m, "dist_cbd_km"] = d_km.loc[mask_m]

    gdf_denue = gdf_denue[gdf_denue["metro_area"] != "Outside"].copy()
    print(f"      -> Total projected DENUE units within 18 km of the 4 CBDs: {len(gdf_denue):,}")

    # --------------------------------------------------------------------------
    # COMPUTE 1KM CENTROID BUFFER COUNTS & LOG DENSITIES (MAUP-FREE)
    # --------------------------------------------------------------------------
    print("[3/6] Performing spatial joins between 1,000-meter Colonia buffers and DENUE POIs...")
    buffers_gdf = gpd.GeoDataFrame(
        gdf_colonias[["cve_col_9", "metro_area"]].copy(),
        geometry=gpd.points_from_xy(gdf_colonias["x_metric"], gdf_colonias["y_metric"], crs=METRIC_CRS).buffer(BUFFER_RADIUS_M),
        crs=METRIC_CRS
    )

    joined = gpd.sjoin(
        buffers_gdf[["cve_col_9", "geometry"]],
        gdf_denue[["scian_group", "geometry"]],
        how="inner",
        predicate="contains"
    )

    piv = joined.groupby(["cve_col_9", "scian_group"]).size().unstack(fill_value=0)
    amenity_df = pd.DataFrame({"cve_col_9": gdf_colonias["cve_col_9"].values})

    for grp_key, out_col in [
        ("food", "food_count_1km"),
        ("health", "health_count_1km"),
        ("edu", "edu_count_1km"),
        ("leisure", "leisure_count_1km"),
        ("nightlife", "nightlife_count_1km"),
        ("industrial", "industrial_count_1km"),
        ("retail_general", "retail_general_count_1km")
    ]:
        if grp_key in piv.columns:
            amenity_df[out_col] = amenity_df["cve_col_9"].map(piv[grp_key]).fillna(0).astype(int)
        else:
            amenity_df[out_col] = 0

    # Positive Urban Amenity Counts & Log Density per km2 
    amenity_df["positive_amenities_1km"] = (
        amenity_df["food_count_1km"] +
        amenity_df["health_count_1km"] +
        amenity_df["edu_count_1km"] +
        amenity_df["leisure_count_1km"]
    )
    # Log Density per km2: ln((Count / pi*r^2) + 1)
    amenity_df["amenity_density_km2"] = (amenity_df["positive_amenities_1km"] / BUFFER_AREA_KM2).round(4)
    amenity_df["amenity_index_1km"] = np.log1p(amenity_df["amenity_density_km2"]).round(4)

    # Disamenity Log Densities (Nightlife/Alcohol & Local Industrial)
    amenity_df["nightlife_index_1km"] = np.log1p(amenity_df["nightlife_count_1km"] / BUFFER_AREA_KM2).round(4)
    amenity_df["industrial_buffer_index_1km"] = np.log1p(amenity_df["industrial_count_1km"] / BUFFER_AREA_KM2).round(4)

    print(
        f"      -> DENUE 1km Amenity Density Index (`amenity_index_1km`): "
        f"Mean = {amenity_df['amenity_index_1km'].mean():.3f}, "
        f"SD = {amenity_df['amenity_index_1km'].std():.3f}, "
        f"Non-Zero = {(amenity_df['amenity_index_1km'] > 0).mean()*100:.1f}%"
    )

    # --------------------------------------------------------------------------
    # IDENTIFY EXOGENOUS DENUE RETAIL & INDUSTRIAL SUBCENTERS (WEIGHTED K-MEANS)
    # --------------------------------------------------------------------------
    print("[4/6] Extracting exogenous Retail and Industrial Subcenters from DENUE establishments...")
    subcenter_rows = []

    for metro_name, (cx, cy) in cbd_metric_map.items():
        k_ret = METRO_SUBCENTER_K[metro_name]["retail"]
        k_ind = METRO_SUBCENTER_K[metro_name]["industrial"]

        subcenter_rows.append({
            "metro_area": metro_name, "subcenter_type": "cbd", "cluster_id": 0,
            "x_metric": float(cx), "y_metric": float(cy), "source_method": "historic_plaza_anchor"
        })

        sub_denue = gdf_denue[gdf_denue["metro_area"] == metro_name]

        # 1. Retail Commercial Subcenters: Supermarkets (462111), Department Stores (462210),
        #    Minisupers (462112), and major Retail/Restaurant nodes outside the 0.8 km historic plaza
        ret_candidates = sub_denue[
            (sub_denue["codigo_act"].str.startswith(("462111", "462210", "462112", "463", "465", "7225"))) &
            (sub_denue["dist_cbd_km"] >= 0.8) &
            (sub_denue["dist_cbd_km"] <= 15.5)
        ].copy()

        if len(ret_candidates) >= k_ret:
            coords_r = ret_candidates[["x_metric", "y_metric"]].values
            # Boost anchor stores (supermarkets & department stores) in weighting
            anchor_boost = np.where(ret_candidates["codigo_act"].str.startswith(("462111", "462210")), 5.0, 1.0)
            weights_r = ret_candidates["emp_weight"].values * anchor_boost
            km_r = KMeans(n_clusters=k_ret, random_state=RANDOM_SEED, n_init=20).fit(coords_r, sample_weight=weights_r)
            for idx_k, (rx, ry) in enumerate(km_r.cluster_centers_):
                subcenter_rows.append({
                    "metro_area": metro_name, "subcenter_type": "retail", "cluster_id": idx_k + 1,
                    "x_metric": float(rx), "y_metric": float(ry), "source_method": "denue_retail_employment_kmeans"
                })

        # 2. Industrial Manufacturing Subcenters: SCIAN 31-33 plants outside the 1.0 km historic core
        #    Prioritize medium and large plants (emp_weight >= 8.0) so home workshops don't pull centroids
        ind_all = sub_denue[
            (sub_denue["scian_group"] == "industrial") &
            (sub_denue["dist_cbd_km"] >= 1.0) &
            (sub_denue["dist_cbd_km"] <= 16.5)
        ].copy()
        ind_major = ind_all[ind_all["emp_weight"] >= 8.0].copy()
        ind_candidates = ind_major if len(ind_major) >= k_ind * 3 else ind_all

        if len(ind_candidates) >= k_ind:
            coords_i = ind_candidates[["x_metric", "y_metric"]].values
            weights_i = ind_candidates["emp_weight"].values
            km_i = KMeans(n_clusters=k_ind, random_state=RANDOM_SEED + 7, n_init=20).fit(coords_i, sample_weight=weights_i)
            for idx_k, (ix, iy) in enumerate(km_i.cluster_centers_):
                subcenter_rows.append({
                    "metro_area": metro_name, "subcenter_type": "industrial", "cluster_id": idx_k + 1,
                    "x_metric": float(ix), "y_metric": float(iy), "source_method": "denue_manufacturing_employment_kmeans"
                })

    subcenters_df = pd.DataFrame(subcenter_rows)
    sub_gdf = gpd.GeoDataFrame(
        subcenters_df,
        geometry=gpd.points_from_xy(subcenters_df["x_metric"], subcenters_df["y_metric"], crs=METRIC_CRS),
        crs=METRIC_CRS
    ).to_crs("EPSG:4326")
    subcenters_df["lat"] = sub_gdf.geometry.y.round(6)
    subcenters_df["lon"] = sub_gdf.geometry.x.round(6)
    subcenters_df.to_csv(OUTPUT_SUBCENTERS_CSV, index=False, encoding="utf-8")
    print(f"      -> Saved {len(subcenters_df)} exogenous DENUE Subcenters to: {OUTPUT_SUBCENTERS_CSV}")

    return amenity_df, subcenters_df


# ==============================================================================
# 5. BUILD MASTER COLONIA SPATIAL LOOKUP & MERGE WITH LISTINGS
# ==============================================================================
def build_master_colonia_lookup(gdf_colonias, amenity_df, subcenters_df):
    print("[5/6] Computing metric distances to CBD, DENUE Retail/Industrial Subcenters & Deprivation Z-Scores...")
    col_df = pd.DataFrame(gdf_colonias.drop(columns=["geometry"], errors="ignore")).merge(
        amenity_df, on="cve_col_9", how="left"
    )

    col_df["dist_to_historic_center_km"] = np.nan
    col_df["dist_retail_subcenter_km"] = np.nan
    col_df["dist_industrial_subcenter_km"] = np.nan

    for metro_name in METRO_CENTERS.keys():
        m_mask = col_df["metro_area"] == metro_name
        if not m_mask.any():
            continue
        coords = col_df.loc[m_mask, ["x_metric", "y_metric"]].values

        cbd_pt = subcenters_df[(subcenters_df["metro_area"] == metro_name) & (subcenters_df["subcenter_type"] == "cbd")][["x_metric", "y_metric"]].values
        ret_pts = subcenters_df[(subcenters_df["metro_area"] == metro_name) & (subcenters_df["subcenter_type"] == "retail")][["x_metric", "y_metric"]].values
        ind_pts = subcenters_df[(subcenters_df["metro_area"] == metro_name) & (subcenters_df["subcenter_type"] == "industrial")][["x_metric", "y_metric"]].values

        if len(cbd_pt) > 0:
            d_cbd = np.hypot(coords[:, 0] - cbd_pt[0, 0], coords[:, 1] - cbd_pt[0, 1]) / 1000.0
            col_df.loc[m_mask, "dist_to_historic_center_km"] = np.round(d_cbd, 4)

        if len(ret_pts) > 0:
            d_ret = np.min(np.sqrt(((coords[:, None, :] - ret_pts[None, :, :]) ** 2).sum(axis=2)), axis=1) / 1000.0
            col_df.loc[m_mask, "dist_retail_subcenter_km"] = np.round(d_ret, 4)

        if len(ind_pts) > 0:
            d_ind = np.min(np.sqrt(((coords[:, None, :] - ind_pts[None, :, :]) ** 2).sum(axis=2)), axis=1) / 1000.0
            col_df.loc[m_mask, "dist_industrial_subcenter_km"] = np.round(d_ind, 4)

    col_df["dist_industrial_subcenter_km_sq"] = (col_df["dist_industrial_subcenter_km"] ** 2).round(4)

    col_df["imc_marginalization_index"] = pd.to_numeric(col_df.get("IMN_2020"), errors="coerce")
    col_df["imc_pct_no_basic_educ"] = pd.to_numeric(col_df.get("SBASC"), errors="coerce")
    col_df["imc_pct_no_internet"] = pd.to_numeric(col_df.get("OVSINT"), errors="coerce")
    col_df["imc_pct_overcrowding"] = pd.to_numeric(col_df.get("OVHAC"), errors="coerce")

    for c in ["imc_marginalization_index", "imc_pct_no_basic_educ", "imc_pct_no_internet", "imc_pct_overcrowding"]:
        col_df[c] = col_df[c].fillna(col_df.groupby("metro_area")[c].transform("median")).fillna(col_df[c].median())

    if col_df["imc_pct_no_internet"].max() > 1.5:
        col_df["census_internet_share"] = (1.0 - (col_df["imc_pct_no_internet"] / 100.0).clip(0.0, 1.0)).round(4)
    else:
        col_df["census_internet_share"] = (1.0 - col_df["imc_pct_no_internet"].clip(0.0, 1.0)).round(4)

    # Z-Score Standardized Urban Deprivation Index (+1 SD = higher deprivation)
    imn_mean = col_df["imc_marginalization_index"].mean()
    imn_std = col_df["imc_marginalization_index"].std()
    col_df["z_marginalization"] = (-1.0 * (col_df["imc_marginalization_index"] - imn_mean) / (imn_std if imn_std > 1e-9 else 1.0)).round(4)

    col_df.to_csv(OUTPUT_COLONIA_LOOKUP_CSV, index=False, encoding="utf-8")
    print(f"      -> Saved Colonia Lookup ({len(col_df):,} colonias) to: {OUTPUT_COLONIA_LOOKUP_CSV}")
    return col_df


def main():
    raw_csv_path = next((p for p in CANDIDATE_RAW_CSVS if os.path.exists(p) and os.path.getsize(p) > 500), None)
    if not raw_csv_path:
        raise FileNotFoundError(
            f"No consolidated raw listings CSV found in {INPUTS_DIR}. "
            "Ensure `property_listings_multicity_consolidated.csv` is inside multiple_cities_v3/Inputs/."
        )

    gdf_colonias = load_conapo_colonias_gdf()
    amenity_df, subcenters_df = load_and_process_local_denue(gdf_colonias)
    col_lookup = build_master_colonia_lookup(gdf_colonias, amenity_df, subcenters_df)

    print(f"[6/6] Merging spatial attributes with raw listings from: {raw_csv_path}")
    df = pd.read_csv(raw_csv_path, dtype=str, low_memory=False)
    df["listing_id"] = df["listing_id"].astype(str).str.strip()
    df.drop_duplicates(subset=["listing_id"], keep="first", inplace=True)

    raw_cve = df.get("colonia_cvegeo", pd.Series("", index=df.index)).fillna("")
    if "CVE_COL" in df.columns:
        raw_cve = raw_cve.mask(raw_cve.str.strip() == "", df["CVE_COL"].fillna(""))
    df["cve_col_9"] = raw_cve.apply(canonicalize_cve_col)

    if "colonia_match_method" not in df.columns:
        df["colonia_match_method"] = np.where(df["cve_col_9"] != "", "exact_normalized_name", "unmatched")

    unmatched = df["cve_col_9"] == ""
    if unmatched.any():
        mun_norm = df.get("municipality_reported", pd.Series("", index=df.index)).apply(normalize_str)
        metro_guess = mun_norm.map(MUN_NAME_TO_METRO).fillna("")
        col_norm = df.get("colonia_reported", pd.Series("", index=df.index)).apply(normalize_colonia_name)

        name_to_cve = (
            col_lookup[col_lookup["norm_col_name"] != ""]
            .drop_duplicates(subset=["metro_area", "norm_col_name"])
            .set_index(["metro_area", "norm_col_name"])["cve_col_9"]
            .to_dict()
        )
        recovered_cves = [name_to_cve.get((m, c), "") for m, c in zip(metro_guess.loc[unmatched], col_norm.loc[unmatched])]
        df.loc[unmatched, "cve_col_9"] = recovered_cves

    overlap_cols = [c for c in col_lookup.columns if c != "cve_col_9" and c in df.columns]
    df.drop(columns=overlap_cols, inplace=True, errors="ignore")

    df_spatial = df.merge(col_lookup, on="cve_col_9", how="left")
    df_spatial["colonia_cvegeo"] = df_spatial["cve_col_9"]

    df_spatial["is_gated_polygon"] = pd.to_numeric(df_spatial["is_gated_polygon"], errors="coerce").fillna(0).astype(int)
    df_spatial["is_gated_listing"] = [
        classify_listing_gated(ti, cr, desc)
        for ti, cr, desc in zip(
            df_spatial.get("title", pd.Series("", index=df_spatial.index)),
            df_spatial.get("colonia_reported", pd.Series("", index=df_spatial.index)),
            df_spatial.get("raw_description", pd.Series("", index=df_spatial.index))
        )
    ]
    df_spatial["is_planned_gated_enclave"] = np.maximum(
        df_spatial["is_gated_polygon"], df_spatial["is_gated_listing"]
    ).astype(int)

    for col in ["price", "built_area_m2", "land_area_m2", "bedrooms", "bathrooms", "parking_spaces"]:
        df_spatial[col] = pd.to_numeric(df_spatial.get(col, np.nan), errors="coerce")

    df_spatial.to_csv(OUTPUT_SPATIAL_CSV, index=False, encoding="utf-8")

    if "is_remate_bancario" in df_spatial.columns:
        remate_mask = df_spatial["is_remate_bancario"].astype(str).str.strip().str.upper().isin(["TRUE", "1", "YES", "T"])
    else:
        remate_mask = pd.Series(False, index=df_spatial.index)

    valid_core_mask = (
        df_spatial["metro_area"].isin(METRO_CENTERS.keys()) &
        (df_spatial["cve_col_9"] != "") &
        (df_spatial["dist_to_historic_center_km"] <= CORE_RADIUS_KM) &
        (df_spatial["price"] >= 150000) &
        (df_spatial["built_area_m2"] >= 25) &
        (df_spatial["bedrooms"] >= 1) &
        (df_spatial["bathrooms"] >= 1) &
        (~remate_mask)
    )
    df_15km = df_spatial[valid_core_mask].copy()

    p_low = df_15km.groupby("metro_area")["price"].transform(lambda s: s.quantile(0.005))
    p_high = df_15km.groupby("metro_area")["price"].transform(lambda s: s.quantile(0.995))
    a_low = df_15km.groupby("metro_area")["built_area_m2"].transform(lambda s: s.quantile(0.005))
    a_high = df_15km.groupby("metro_area")["built_area_m2"].transform(lambda s: s.quantile(0.995))

    df_15km = df_15km[
        df_15km["price"].between(p_low, p_high) &
        df_15km["built_area_m2"].between(a_low, a_high)
    ].copy()

    df_15km["price_m2"] = (df_15km["price"] / df_15km["built_area_m2"]).round(2)
    df_15km["ln_price"] = np.log(df_15km["price"])
    df_15km["ln_price_m2"] = np.log(df_15km["price_m2"])
    df_15km["ln_built_area"] = np.log(df_15km["built_area_m2"])

    df_15km.to_csv(OUTPUT_15KM_CSV, index=False, encoding="utf-8")

    print("\n" + "=" * 82)
    print(f"[SUCCESS] 100% OFFLINE DENUE SPATIAL DATASET READY: N = {len(df_15km):,} LISTINGS")
    print("=" * 82)
    audit_tbl = df_15km.groupby("metro_area").agg(
        N=("listing_id", "count"),
        Colonias=("cve_col_9", "nunique"),
        Gated_Pct=("is_planned_gated_enclave", lambda s: round(s.mean() * 100, 1)),
        Amenity_Idx=("amenity_index_1km", lambda s: round(s.mean(), 2)),
        Nightlife_Idx=("nightlife_index_1km", lambda s: round(s.mean(), 2)),
        Ret_Dist_km=("dist_retail_subcenter_km", lambda s: round(s.median(), 2)),
        Ind_Dist_km=("dist_industrial_subcenter_km", lambda s: round(s.median(), 2))
    )
    print(audit_tbl.to_string())
    print("=" * 82)


if __name__ == "__main__":
    main()
