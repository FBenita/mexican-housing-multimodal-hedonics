# -*- coding: utf-8 -*-
"""
02_filter_15km_and_audit.py
Filters raw scraped residential listings to the 15-kilometer urban core boundary,
excludes bank foreclosures (remates bancarios), reconciles 9-digit CONAPO
settlement keys, and exports data quality audit reports.
"""

import re
import difflib
import unicodedata
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

try:
    from rapidfuzz import fuzz, process
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = REPO_ROOT / "data" / "raw"
EXTERNAL_DATA_DIR = REPO_ROOT / "data" / "external"
AUDIT_DIR = REPO_ROOT / "outputs" / "audit"

RAW_CSV = RAW_DATA_DIR / "property_listings_multicity_consolidated.csv"
FILTERED_CSV = RAW_DATA_DIR / "property_listings_multicity_filtered_15km.csv"
SHP_PATH = EXTERNAL_DATA_DIR / "colonias_imc2020_shp" / "colonias_imc2020.shp"

CORE_RADIUS_KM = 15.0
FUZZY_MATCH_THRESHOLD = 85.0

METRO_MAPPING = {
    "AGUASCALIENTES": "Aguascalientes",
    "JESUS MARIA": "Aguascalientes",
    "SAN FRANCISCO DE LOS ROMO": "Aguascalientes",
    "LEON": "Leon",
    "PURISIMA DEL RINCON": "Leon",
    "SAN FRANCISCO DEL RINCON": "Leon",
    "QUERETARO": "Queretaro",
    "CORREGIDORA": "Queretaro",
    "EL MARQUES": "Queretaro",
    "HUIMILPAN": "Queretaro",
    "ZACATECAS": "Zacatecas-Guadalupe",
    "GUADALUPE": "Zacatecas-Guadalupe",
    "MORELOS": "Zacatecas-Guadalupe",
    "VETAGRANDE": "Zacatecas-Guadalupe",
    "TRANCOSO": "Zacatecas-Guadalupe"
}

MUN_TO_METRO_CODES = {
    "AGUASCALIENTES": ["01001", "01005", "01011"],
    "JESUS MARIA": ["01005", "01001", "01011"],
    "SAN FRANCISCO DE LOS ROMO": ["01011", "01001", "01005"],
    "LEON": ["11020", "11025", "11031"],
    "PURISIMA DEL RINCON": ["11025", "11020", "11031"],
    "SAN FRANCISCO DEL RINCON": ["11031", "11020", "11025"],
    "QUERETARO": ["22014", "22006", "22011", "22008"],
    "CORREGIDORA": ["22006", "22014", "22011", "22008"],
    "EL MARQUES": ["22011", "22014", "22006", "22008"],
    "HUIMILPAN": ["22008", "22014", "22006", "22011"],
    "ZACATECAS": ["32056", "32017", "32032", "32050", "32057"],
    "GUADALUPE": ["32017", "32056", "32032", "32050", "32057"],
    "MORELOS": ["32032", "32056", "32017"],
    "VETAGRANDE": ["32050", "32056", "32017"],
    "TRANCOSO": ["32057", "32017", "32056"]
}

ALL_TARGET_MUN_CODES = [
    "01001", "01005", "01011",
    "11020", "11025", "11031",
    "22014", "22006", "22011", "22008",
    "32017", "32032", "32050", "32056", "32057"
]

MUN_CODE_NAMES = {
    "01001": ("Aguascalientes", "Aguascalientes"),
    "01005": ("Aguascalientes", "Jesus Maria"),
    "01011": ("Aguascalientes", "San Francisco de los Romo"),
    "11020": ("Leon", "Leon"),
    "11025": ("Leon", "Purisima del Rincon"),
    "11031": ("Leon", "San Francisco del Rincon"),
    "22014": ("Queretaro", "Queretaro"),
    "22006": ("Queretaro", "Corregidora"),
    "22011": ("Queretaro", "El Marques"),
    "22008": ("Queretaro", "Huimilpan"),
    "32056": ("Zacatecas-Guadalupe", "Zacatecas"),
    "32017": ("Zacatecas-Guadalupe", "Guadalupe"),
    "32032": ("Zacatecas-Guadalupe", "Morelos"),
    "32050": ("Zacatecas-Guadalupe", "Vetagrande"),
    "32057": ("Zacatecas-Guadalupe", "Trancoso")
}

METRO_CBD_COORDS = {
    "Aguascalientes": (21.8807, -102.2961),
    "Leon": (21.1221, -101.6826),
    "Queretaro": (20.5931, -100.3920),
    "Zacatecas-Guadalupe": (22.7746, -102.5737)
}

METRO_ORDER = ["Aguascalientes", "Leon", "Queretaro", "Zacatecas-Guadalupe"]


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, l1, p2, l2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((p2 - p1) / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin((l2 - l1) / 2.0) ** 2
    return 2.0 * R * np.arcsin(np.sqrt(a))


def classify_conapo_tipo(clasif_raw, col_nombre=""):
    t = str(clasif_raw or "").upper().strip()
    n = str(col_nombre or "").upper().strip()
    comb = f"{t} {n}"

    if t in ["FRACCIONAMIENTO", "CONDOMINIO", "RESIDENCIAL", "CONJUNTO HABITACIONAL", "PRIVADA", "RINCONADA", "VILLA"] or \
       any(k in comb for k in ["PRIVADA", "COTO ", "RESIDENCIAL", "CAMPESTRE", "CLUB DE GOLF", "FRACC"]):
        return "Planned_Gated_Enclave", 1

    if t in ["EJIDO", "RANCHERIA", "RANCHERÍA", "PUEBLO", "GRANJA", "AMPLIACION", "AMPLIACIÓN", "RANCHO", "PARAJE", "EX-HACIENDA", "HACIENDA"]:
        return "PeriUrban_Ejido_Pueblo", 0

    if t in ["UNIDAD HABITACIONAL", "UNIDAD", "ZONA INDUSTRIAL", "PARQUE INDUSTRIAL", "CIUDAD INDUSTRIAL", "CORREDOR INDUSTRIAL"]:
        return "Social_Unidad_Institutional", 0

    return "Traditional_Open_Colonia", 0


def normalize_text(text):
    if text is None or pd.isna(text):
        return ""
    s = str(text).strip()
    try:
        if "Ã" in s or "Â" in s:
            s = s.encode("latin1").decode("utf-8")
    except Exception:
        pass
    s = s.upper()
    s = "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    prefix_pattern = re.compile(
        r"^(?:RANCHO\s+O\s+RANCHERIA|UNIDAD\s+HABITACIONAL|FRACCIONAMIENTO|FRACC|"
        r"COLONIA|COL|PRIVADA|PRIV|RESIDENCIAL|RES|BARRIO|BARR|CONJUNTO\s+HABITACIONAL|"
        r"CONDOMINIO|COND|COTO|SECTOR|AMPLIACION|AMPL|EJIDO|ZONA|VALLE\s+RESIDENCIAL)\b\s*"
    )
    prev = None
    while s and s != prev:
        prev = s
        s = prefix_pattern.sub("", s).strip()

    s = re.sub(r"\b1(?:RA|ERA|A)?\s+SECC(?:ION)?\b", "PRIMERA SECCION", s)
    s = re.sub(r"\b2(?:DA|A)?\s+SECC(?:ION)?\b", "SEGUNDA SECCION", s)
    s = re.sub(r"\b3(?:RA|ERA|A)?\s+SECC(?:ION)?\b", "TERCERA SECCION", s)
    return re.sub(r"\s+", " ", s).strip()


def match_colonia_scoped(col_str, mun_str, gdf_by_mun):
    norm_q = normalize_text(col_str)
    mun_key = normalize_text(mun_str)
    if not norm_q or not mun_key or mun_key not in MUN_TO_METRO_CODES:
        return "unmatched", np.nan, "", "", "", ""

    mun_codes = MUN_TO_METRO_CODES[mun_key]
    primary_mun = mun_codes[0]

    if norm_q in ("CENTRO", "ZONA CENTRO", "CENTRO HISTORICO", f"{mun_key} CENTRO"):
        sub_p = gdf_by_mun.get(primary_mun, pd.DataFrame())
        if not sub_p.empty:
            c_hits = sub_p[sub_p["norm_name"].isin(["ZONA CENTRO", "CENTRO", "CENTRO HISTORICO"])]
            if not c_hits.empty:
                r0 = c_hits.iloc[0]
                return "exact_normalized_name", 100.0, r0["CVEGEO_str"], r0["COLONIA"], r0["CVE_MUN_FULL"], r0["CVE_ENT_str"]

    for m_code in mun_codes:
        sub_m = gdf_by_mun.get(m_code, pd.DataFrame())
        if sub_m.empty:
            continue
        ex = sub_m[sub_m["norm_name"] == norm_q]
        if not ex.empty:
            r0 = ex.iloc[0]
            return "exact_normalized_name", 100.0, r0["CVEGEO_str"], r0["COLONIA"], r0["CVE_MUN_FULL"], r0["CVE_ENT_str"]

    for m_code in mun_codes:
        sub_m = gdf_by_mun.get(m_code, pd.DataFrame())
        if sub_m.empty:
            continue

        if HAS_RAPIDFUZZ:
            choices = sub_m["norm_name"].tolist()
            match_res = process.extractOne(norm_q, choices, scorer=fuzz.token_set_ratio)
            if match_res and match_res[1] >= FUZZY_MATCH_THRESHOLD:
                _, score, idx = match_res
                r0 = sub_m.iloc[idx]
                return "fuzzy_name_match", round(float(score), 2), r0["CVEGEO_str"], r0["COLONIA"], r0["CVE_MUN_FULL"], r0["CVE_ENT_str"]
        else:
            best_sc, best_r = 0.0, None
            for _, r_cand in sub_m.iterrows():
                t1, t2 = set(norm_q.split()), set(r_cand["norm_name"].split())
                if not t1 or not t2:
                    continue
                inter = t1 & t2
                s_inter = " ".join(sorted(inter))
                if (inter == t1 or inter == t2) and len(s_inter) >= 4:
                    sc = 100.0 * (len(s_inter) / max(min(len(norm_q), len(r_cand["norm_name"])), 1))
                else:
                    comb1 = " ".join(sorted(inter) + sorted(t1 - inter))
                    comb2 = " ".join(sorted(inter) + sorted(t2 - inter))
                    sc = 100.0 * max(
                        difflib.SequenceMatcher(None, s_inter, comb1).ratio(),
                        difflib.SequenceMatcher(None, s_inter, comb2).ratio(),
                        difflib.SequenceMatcher(None, comb1, comb2).ratio(),
                        difflib.SequenceMatcher(None, norm_q, r_cand["norm_name"]).ratio()
                    )
                if sc > best_sc:
                    best_sc, best_r = sc, r_cand
            if best_r is not None and best_sc >= FUZZY_MATCH_THRESHOLD:
                return "fuzzy_name_match", round(float(best_sc), 2), best_r["CVEGEO_str"], best_r["COLONIA"], best_r["CVE_MUN_FULL"], best_r["CVE_ENT_str"]

    return "unmatched", np.nan, "", "", "", ""


def main():
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RAW_CSV, low_memory=False)
    try:
        gdf = gpd.read_file(SHP_PATH)
    except UnicodeDecodeError:
        gdf = gpd.read_file(SHP_PATH, encoding="latin1")

    if gdf.crs is None:
        gdf.set_crs("EPSG:6372", inplace=True)
    gdf_metric = gdf.to_crs("EPSG:6372")
    centroids_wgs = gdf_metric.geometry.centroid.to_crs("EPSG:4326")
    gdf["centroid_lat"] = centroids_wgs.y
    gdf["centroid_lon"] = centroids_wgs.x

    gdf["CVE_MUN_FULL"] = gdf["CVE_MUN"].astype(str).str.strip().str.replace(r"\.0+$", "", regex=True).str.zfill(5)
    gdf["CVE_ENT_str"] = gdf["CVE_ENT"].astype(str).str.strip().str.replace(r"\.0+$", "", regex=True).str.zfill(2)

    if "CVE_COL" in gdf.columns:
        gdf["CVEGEO_str"] = gdf["CVE_COL"].astype(str).str.strip().str.replace("_", "", regex=False)
    else:
        gdf["CVEGEO_str"] = gdf["CVE_MUN_FULL"] + gdf.index.astype(str).str.zfill(4)

    gdf_target = gdf[gdf["CVE_MUN_FULL"].isin(ALL_TARGET_MUN_CODES)].copy()
    gdf_target["norm_name"] = gdf_target["COLONIA"].apply(normalize_text)

    for col_needed, default_v in [("CLASIF", "Colonia"), ("IMN_2020", np.nan), ("GM_2020", ""), ("POBTOT", np.nan)]:
        if col_needed not in gdf_target.columns:
            gdf_target[col_needed] = default_v

    df["mun_norm"] = df["municipality_reported"].apply(normalize_text)
    df["metro_area"] = df["mun_norm"].map(METRO_MAPPING).fillna("Unmapped")

    df["colonia_cvegeo_clean"] = (
        df["colonia_cvegeo"].fillna("").astype(str).str.strip()
        .str.replace(r"\.0+$", "", regex=True).str.replace("_", "", regex=False)
    )
    df["colonia_cvegeo_clean"] = df["colonia_cvegeo_clean"].apply(lambda x: x.zfill(9) if len(x) >= 7 else "")

    valid_cve_set = set(gdf_target["CVEGEO_str"].unique())
    already_matched_ratio = df["colonia_cvegeo_clean"].isin(valid_cve_set).mean()

    if already_matched_ratio < 0.50:
        gdf_by_mun = {m: gdf_target[gdf_target["CVE_MUN_FULL"] == m].copy() for m in ALL_TARGET_MUN_CODES}
        new_methods, new_scores, new_cves, new_noms = [], [], [], []
        for _, r in df.iterrows():
            m_meth, m_score, cvegeo, col_nom, _, _ = match_colonia_scoped(
                r.get("colonia_reported", ""), r.get("municipality_reported", ""), gdf_by_mun
            )
            new_methods.append(m_meth)
            new_scores.append(m_score)
            new_cves.append(cvegeo)
            new_noms.append(col_nom)

        df["colonia_match_method"] = new_methods
        df["colonia_match_score"] = new_scores
        df["colonia_cvegeo"] = new_cves
        df["colonia_nombre"] = new_noms
        df["colonia_cvegeo_clean"] = df["colonia_cvegeo"].fillna("").astype(str).str.strip()
        df.to_csv(RAW_CSV, index=False, encoding="utf-8")

    cve_lookup = (
        gdf_target.drop_duplicates(subset=["CVEGEO_str"])
        .set_index("CVEGEO_str")[["centroid_lat", "centroid_lon", "CVE_MUN_FULL", "CLASIF", "IMN_2020", "GM_2020", "POBTOT"]]
        .to_dict(orient="index")
    )

    lats, lons, cve_muns, tipos, imns, gms, pobs = [], [], [], [], [], [], []
    for cve in df["colonia_cvegeo_clean"]:
        info = cve_lookup.get(cve, None)
        if info:
            lats.append(info["centroid_lat"])
            lons.append(info["centroid_lon"])
            cve_muns.append(info["CVE_MUN_FULL"])
            tipos.append(info["CLASIF"])
            imns.append(info["IMN_2020"])
            gms.append(info["GM_2020"])
            pobs.append(info["POBTOT"])
        else:
            lats.append(np.nan)
            lons.append(np.nan)
            cve_muns.append("")
            tipos.append("")
            imns.append(np.nan)
            gms.append("")
            pobs.append(np.nan)

    df["centroid_lat"] = lats
    df["centroid_lon"] = lons
    df["CVE_MUN_FULL"] = cve_muns
    df["conapo_clasif"] = tipos
    df["conapo_imn2020"] = pd.to_numeric(imns, errors="coerce")
    df["conapo_gm2020"] = gms
    df["conapo_pobtot"] = pd.to_numeric(pobs, errors="coerce")

    morph_res = df.apply(
        lambda r: classify_conapo_tipo(r.get("conapo_clasif", ""), f"{r.get('colonia_nombre', '')} {r.get('colonia_reported', '')}"),
        axis=1
    )
    df["morphology_class"] = [x[0] for x in morph_res]
    df["morph_planned_gated"] = [x[1] for x in morph_res]

    df["dist_to_historic_center_km"] = np.nan
    for metro, (c_lat, c_lon) in METRO_CBD_COORDS.items():
        m_mask = (df["metro_area"] == metro) & df["centroid_lat"].notna() & df["centroid_lon"].notna()
        if m_mask.any():
            df.loc[m_mask, "dist_to_historic_center_km"] = haversine_km(
                df.loc[m_mask, "centroid_lat"].values,
                df.loc[m_mask, "centroid_lon"].values,
                c_lat, c_lon
            ).round(3)

    for col in ["price", "built_area_m2", "land_area_m2", "bedrooms", "bathrooms", "parking_spaces"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    valid_hedonic_mask = (
        (df["metro_area"] != "Unmapped") &
        (df["colonia_cvegeo_clean"] != "") &
        (df["dist_to_historic_center_km"] <= CORE_RADIUS_KM) &
        (df["price"] >= 200000) & (df["price"] <= 35000000) &
        (df["built_area_m2"] >= 30) & (df["built_area_m2"] <= 1500) &
        (df["bedrooms"] >= 1) & (df["bedrooms"] <= 12) &
        (df["bathrooms"] >= 1) & (df["bathrooms"] <= 12) &
        (~df["is_remate_bancario"].astype(bool))
    )
    df_filtered = df[valid_hedonic_mask].copy()
    df_filtered["price_per_m2"] = (df_filtered["price"] / df_filtered["built_area_m2"]).round(2)
    df_filtered.to_csv(FILTERED_CSV, index=False, encoding="utf-8")
    print(f"[SUCCESS] Saved {len(df_filtered)} filtered core listings (<= {CORE_RADIUS_KM} km) to {FILTERED_CSV}")


if __name__ == "__main__":
    main()