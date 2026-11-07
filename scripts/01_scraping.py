# -*- coding: utf-8 -*-
"""
01_scraping.py
Consolidated Multi-City Web Scraper & Municipality-Scoped CONAPO Colonia Matcher
Target Metropolises (Flagship 4-City Matrix):
  - Aguascalientes: Aguascalientes (01001), Jesús María (01005), San Francisco de los Romo (01011)
  - León: León (11020), Purísima del Rincón (11025), San Francisco del Rincón (11031)
  - Querétaro: Querétaro (22014), Corregidora (22006), El Marqués (22011), Huimilpan (22008)
  - Zacatecas-Guadalupe: Zacatecas (32056), Guadalupe (32017), Morelos (32032), Vetagrande (32050), Trancoso (32057)
"""

import os
import re
import json
import time
import random
import hashlib
import unicodedata
import difflib
from datetime import datetime
from urllib.parse import urljoin, urlparse, urlunparse, parse_qs, urlencode
import warnings

import pandas as pd
import geopandas as gpd
import requests
from bs4 import BeautifulSoup
warnings.filterwarnings("ignore")

try:
    from rapidfuzz import fuzz, process
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False

BASE_DIR = r"C:\Users"
INPUTS_DIR = os.path.join(BASE_DIR, "Inputs")
DATA_DIR = os.path.join(BASE_DIR, "Data")
SHP_PATH = os.path.join(DATA_DIR, "colonias_imc2020_shp", "colonias_imc2020.shp")
OUTPUT_CSV = os.path.join(INPUTS_DIR, "property_listings_multicity_consolidated.csv")

MAX_PAGES_PER_SOURCE = 30
MIN_DELAY_SECONDS = 2.5
MAX_DELAY_SECONDS = 5.0
FUZZY_MATCH_THRESHOLD = 85.0

# Official Municipalities to scan text for
TARGET_MUNICIPALITIES = {
    "AGUASCALIENTES": "Aguascalientes", "JESUS MARIA": "Aguascalientes", "SAN FRANCISCO DE LOS ROMO": "Aguascalientes",
    "LEON": "Guanajuato", "PURISIMA DEL RINCON": "Guanajuato", "SAN FRANCISCO DEL RINCON": "Guanajuato",
    "QUERETARO": "Queretaro", "CORREGIDORA": "Queretaro", "EL MARQUES": "Queretaro", "HUIMILPAN": "Queretaro",
    "ZACATECAS": "Zacatecas", "GUADALUPE": "Zacatecas", "MORELOS": "Zacatecas", "VETAGRANDE": "Zacatecas", "TRANCOSO": "Zacatecas"
}

MUN_TO_METRO_CODES = {
    "AGUASCALIENTES": ["01001", "01005", "01011"], "JESUS MARIA": ["01005", "01001", "01011"], "SAN FRANCISCO DE LOS ROMO": ["01011", "01001", "01005"],
    "LEON": ["11020", "11025", "11031"], "PURISIMA DEL RINCON": ["11025", "11020", "11031"], "SAN FRANCISCO DEL RINCON": ["11031", "11020", "11025"],
    "QUERETARO": ["22014", "22006", "22011", "22008"], "CORREGIDORA": ["22006", "22014", "22011", "22008"], 
    "EL MARQUES": ["22011", "22014", "22006", "22008"], "HUIMILPAN": ["22008", "22014", "22006", "22011"],
    "ZACATECAS": ["32056", "32017", "32032", "32050", "32057"], "GUADALUPE": ["32017", "32056", "32032", "32050", "32057"],
    "MORELOS": ["32032", "32056", "32017"], "VETAGRANDE": ["32050", "32056", "32017"], "TRANCOSO": ["32057", "32017", "32056"]
}

ALL_TARGET_MUN_CODES = [
    "01001", "01005", "01011",
    "11020", "11025", "11031",
    "22014", "22006", "22011", "22008",
    "32017", "32032", "32050", "32056", "32057"
]

SEARCH_URLS = {
    "icasas": [
        "https://www.icasas.mx/venta/habitacionales-casas-aguascalientes-aguascalientes-2_5_1_0_1_0",
        "https://www.icasas.mx/venta/habitacionales-casas-aguascalientes-jesus-maria",
        "https://www.icasas.mx/venta/habitacionales-casas-guanajuato-leon-2_5_11_0_584_0",
        "https://www.icasas.mx/venta/habitacionales-casas-queretaro-queretaro-2_5_22_0_1349_0",
        "https://www.icasas.mx/venta/habitacionales-casas-queretaro-corregidora",
        "https://www.icasas.mx/venta/habitacionales-casas-queretaro-el-marques",
        "https://www.icasas.mx/venta/habitacionales-casas-zacatecas-zacatecas-2_5_32_0_2451_0",
        "https://www.icasas.mx/venta/habitacionales-casas-zacatecas-guadalupe-2_5_32_0_2410_0"
    ],
    "nocnok": [
        "https://inmuebles.nocnok.com/categoria/casa-en-venta/aguascalientes/aguascalientes",
        "https://inmuebles.nocnok.com/categoria/casa-en-venta/aguascalientes/jesus-maria",
        "https://inmuebles.nocnok.com/categoria/casa-en-venta/aguascalientes/san-francisco-de-los-romo",
        "https://inmuebles.nocnok.com/categoria/casa-en-venta/guanajuato/leon",
        "https://inmuebles.nocnok.com/categoria/casa-en-venta/guanajuato/purisima-del-rincon",
        "https://inmuebles.nocnok.com/categoria/casa-en-venta/guanajuato/san-francisco-del-rincon",
        "https://inmuebles.nocnok.com/categoria/casa-en-venta/queretaro/queretaro",
        "https://inmuebles.nocnok.com/categoria/casa-en-venta/queretaro/el-marques",
        "https://inmuebles.nocnok.com/categoria/casa-en-venta/queretaro/corregidora",
        "https://inmuebles.nocnok.com/categoria/casa-en-venta/zacatecas/zacatecas",
        "https://inmuebles.nocnok.com/categoria/casa-en-venta/zacatecas/guadalupe"
    ]
}

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
REQUEST_TIMEOUT = 25

OUTPUT_COLUMNS = [
    "listing_id", "source", "url_original", "scraped_at", "title", "operation",
    "property_type", "price", "currency", "land_area_m2", "built_area_m2",
    "bedrooms", "bathrooms", "parking_spaces", "street", "colonia_reported",
    "municipality_reported", "state_reported", "postal_code", "is_remate_bancario",
    "colonia_match_method", "colonia_match_score", "colonia_cvegeo", "colonia_nombre",
    "municipality_code", "state_code", "conapo_clasif", "conapo_imn2020",
    "raw_description", "parser_status", "notes"
]


def fix_encoding_mojibake(text):
    if text is None or pd.isna(text): return ""
    s = str(text)
    if "Ã" in s or "Â" in s:
        try: s = s.encode("latin1").decode("utf-8")
        except: pass
    return s

def normalize_text(text):
    if text is None or pd.isna(text): return ""
    s = fix_encoding_mojibake(text).strip().upper()
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

def token_set_ratio_fallback(s1, s2):
    t1, t2 = set(s1.split()), set(s2.split())
    if not t1 or not t2: return 0.0
    inter = t1 & t2
    s_inter = " ".join(sorted(inter))
    if (inter == t1 or inter == t2) and len(s_inter) >= 4:
        return 100.0 * (len(s_inter) / max(min(len(s1), len(s2)), 1))
    comb1 = " ".join(sorted(inter) + sorted(t1 - inter))
    comb2 = " ".join(sorted(inter) + sorted(t2 - inter))
    return 100.0 * max(
        difflib.SequenceMatcher(None, s_inter, comb1).ratio(),
        difflib.SequenceMatcher(None, s_inter, comb2).ratio(),
        difflib.SequenceMatcher(None, comb1, comb2).ratio(),
        difflib.SequenceMatcher(None, s1, s2).ratio()
    )


def load_conapo_shapefile(shp_path):
    print(f"[INFO] Loading CONAPO Colonias Shapefile from {shp_path} ...")
    try:
        gdf = gpd.read_file(shp_path)
    except UnicodeDecodeError:
        gdf = gpd.read_file(shp_path, encoding="latin1")

    gdf["CVE_ENT_str"] = gdf["CVE_ENT"].astype(str).str.strip().str.zfill(2)
    gdf["CVE_MUN_str"] = gdf["CVE_MUN"].astype(str).str.strip().str.zfill(3)
    gdf["CVE_MUN_FULL"] = gdf["CVE_ENT_str"] + gdf["CVE_MUN_str"]
    
    # Generate unique CVEGEO padding based on CVE_COL if missing
    if "CVE_COL" in gdf.columns:
        gdf["CVEGEO_str"] = gdf["CVE_COL"].astype(str).str.replace("_", "")
    else:
        gdf["CVEGEO_str"] = gdf["CVE_MUN_FULL"] + gdf.index.astype(str).str.zfill(4)

    gdf = gdf[gdf["CVE_MUN_FULL"].isin(ALL_TARGET_MUN_CODES)].copy()
    gdf["norm_name"] = gdf["COLONIA"].apply(normalize_text)
    gdf = gdf[(gdf["norm_name"] != "") & (~gdf["norm_name"].isin(["NINGUNO", "SIN NOMBRE", "NA"]))].copy()
    
    # Ensure CLASIF and IMN_2020 are available
    if "CLASIF" not in gdf.columns: gdf["CLASIF"] = "Colonia"
    if "IMN_2020" not in gdf.columns: gdf["IMN_2020"] = ""
    
    print(f"[INFO] Retained {len(gdf)} valid CONAPO colonias across target metropolises.")
    return gdf


def match_colonia_scoped(col_str, mun_str, gdf_by_mun):
    norm_q = normalize_text(col_str)
    mun_key = normalize_text(mun_str)
    empty = ("unmatched", np.nan, "", "", "", "", "", "")
    
    if not norm_q or not mun_key or mun_key not in MUN_TO_METRO_CODES:
        return empty

    mun_codes = MUN_TO_METRO_CODES[mun_key]
    primary_mun = mun_codes[0]

    if norm_q in ("CENTRO", "ZONA CENTRO", "CENTRO HISTORICO", f"{mun_key} CENTRO"):
        sub_p = gdf_by_mun.get(primary_mun, pd.DataFrame())
        if not sub_p.empty:
            c_hits = sub_p[sub_p["norm_name"].isin(["ZONA CENTRO", "CENTRO", "CENTRO HISTORICO"])]
            if not c_hits.empty:
                r0 = c_hits.iloc[0]
                return "exact_normalized_name", 100.0, r0["CVEGEO_str"], r0["COLONIA"], r0["CVE_MUN_str"], r0["CVE_ENT_str"], r0["CLASIF"], r0["IMN_2020"]

    for m_code in mun_codes:
        sub_m = gdf_by_mun.get(m_code, pd.DataFrame())
        if sub_m.empty: continue
        ex = sub_m[sub_m["norm_name"] == norm_q]
        if not ex.empty:
            r0 = ex.iloc[0]
            return "exact_normalized_name", 100.0, r0["CVEGEO_str"], r0["COLONIA"], r0["CVE_MUN_str"], r0["CVE_ENT_str"], r0["CLASIF"], r0["IMN_2020"]

    for m_code in mun_codes:
        sub_m = gdf_by_mun.get(m_code, pd.DataFrame())
        if sub_m.empty: continue
        
        if HAS_RAPIDFUZZ:
            choices = sub_m["norm_name"].tolist()
            match_res = process.extractOne(norm_q, choices, scorer=fuzz.token_set_ratio)
            if match_res and match_res[1] >= FUZZY_MATCH_THRESHOLD:
                _, score, idx = match_res
                r0 = sub_m.iloc[idx]
                return "fuzzy_name_match", round(float(score), 2), r0["CVEGEO_str"], r0["COLONIA"], r0["CVE_MUN_str"], r0["CVE_ENT_str"], r0["CLASIF"], r0["IMN_2020"]
        else:
            best_sc, best_r = 0.0, None
            for _, r_cand in sub_m.iterrows():
                sc = token_set_ratio_fallback(norm_q, r_cand["norm_name"])
                if sc > best_sc:
                    best_sc, best_r = sc, r_cand
            if best_r is not None and best_sc >= FUZZY_MATCH_THRESHOLD:
                return "fuzzy_name_match", round(float(best_sc), 2), best_r["CVEGEO_str"], best_r["COLONIA"], best_r["CVE_MUN_str"], best_r["CVE_ENT_str"], best_r["CLASIF"], best_r["IMN_2020"]

    return empty


def parse_price_from_text(raw_text, json_price=None, json_currency=None):
    price_val, currency_val = None, ""
    if json_price and str(json_price).strip():
        try:
            val = float(re.sub(r"[^\d.]", "", str(json_price)))
            if 150000 <= val <= 80000000: price_val = val
        except ValueError: pass

    if json_currency:
        c_up = str(json_currency).strip().upper()
        currency_val = "USD" if c_up in ("USD", "US$", "DLLS", "DOLARES") else "MXN"
    if price_val and currency_val: return price_val, currency_val

    text = fix_encoding_mojibake(raw_text)
    if not currency_val:
        currency_val = "USD" if re.search(r"\b(?:USD|US\$|D[OÓ]LARES|DLLS)\b", text, re.I) else "MXN"
    if price_val is None:
        text_clean = re.sub(r"\(?\s*\$\s*[\d,]+(?:\.\d+)?\s*/\s*m(?:2|²)\s*\)?", " ", text, flags=re.I)
        patterns = [
            r"(?:US\$|MX\$|\$|MXN|USD|PRECIO\s*[:\-]?\s*\$?)\s*([\d]{1,3}(?:[.,]\d{3})+(?:\.\d{1,2})?|\d{6,9})",
            r"([\d]{1,3}(?:,\d{3})+)\s*(?:MX\$|MXN|PESOS)"
        ]
        for pat in patterns:
            for m in re.finditer(pat, text_clean, flags=re.I):
                raw_num = m.group(1)
                if raw_num.count(".") >= 2: raw_num = raw_num.replace(".", "")
                num = re.sub(r"[^\d.]", "", raw_num.replace(",", ""))
                try:
                    val = float(num)
                    if 150000 <= val <= 80000000: return val, currency_val
                except ValueError: continue
    return price_val, currency_val


def extract_areas_from_text(text):
    s = fix_encoding_mojibake(text)
    land, built = None, None
    def _to_f(n):
        try:
            v = float(n.replace(",", "").replace(" ", "").strip())
            return v if 20 <= v <= 10000 else None
        except: return None

    m_l = re.search(r"(?:terreno|lote|sup\.?\s*terreno)\s*[:\-]?\s*(\d{1,5}(?:[.,]\s?\d{1,3})?)\s*m", s, re.I)
    if m_l: land = _to_f(m_l.group(1))
    m_b = re.search(r"(?:construcci[oó]n|construido|habitable)\s*[:\-]?\s*(\d{1,5}(?:[.,]\s?\d{1,3})?)\s*m", s, re.I)
    if m_b: built = _to_f(m_b.group(1))
    if built is None:
        m_c = re.search(r"\b(\d{2,4}(?:\.\d{1,2})?)\s*(?:m2|m²)\b(?!\s*/)", s, re.I)
        if m_c: built = _to_f(m_c.group(1))
    return land, built

def extract_rooms_and_parking_from_text(text):
    s = fix_encoding_mojibake(text)
    beds, baths, park = None, None, None
    m_bd = re.search(r"(\d{1,2})\s*(?:rec[aá]maras?|habitaciones?|dormitorios?|rec\b)", s, re.I)
    if m_bd: beds = int(m_bd.group(1))
    m_ba = re.search(r"(\d{1,2}(?:\.5)?)\s*(?:ba[ñn]os?\s*completos?|ba[ñn]os?)", s, re.I)
    if m_ba: baths = float(m_ba.group(1))
    m_pk = re.search(r"(\d{1,2})\s*(?:estacionamientos?|cocheras?|autos?|garages?|estac\.)", s, re.I)
    if m_pk: park = int(m_pk.group(1))
    return beds, baths, park

def apply_aggressive_fallback(text, current_built, current_beds, current_baths):
    s = re.sub(r"[\r\n\t]+", " ", fix_encoding_mojibake(text))
    m_nocnok = re.search(r"(\d+)\s*rec\D*?(\d+(?:\.\d+)?)\s*m\D*?(\d+(?:\.\d+)?)\s*ba", s, flags=re.I)
    if m_nocnok:
        if pd.isna(current_beds): current_beds = int(m_nocnok.group(1))
        if pd.isna(current_built): current_built = float(m_nocnok.group(2))
        if pd.isna(current_baths): current_baths = float(m_nocnok.group(3))
    m_icasas = re.search(r"(\d+(?:\.\d+)?)\s*m(?:2|²)\s+(\d+)\s+(\d+(?:\.\d+)?)", s, flags=re.I)
    if m_icasas:
        if pd.isna(current_built): current_built = float(m_icasas.group(1))
        if pd.isna(current_beds): current_beds = int(m_icasas.group(2))
        if pd.isna(current_baths): current_baths = float(m_icasas.group(3))
    return current_built, current_beds, current_baths

def detect_remate_bancario(title, description):
    combined = fix_encoding_mojibake(f"{title or ''} {description or ''}")
    return bool(re.search(
        r"\b(?:remate\s+bancario|remate|recuperaci[oó]n\s+bancaria|adjudicaci[oó]n|"
        r"cesi[oó]n\s+de\s+derechos|litigio|no\s+se\s+aceptan\s+cr[eé]ditos)\b", combined, re.I))

def extract_location_smart(title, raw_description, url):
    title_s = fix_encoding_mojibake(title)
    desc_s = fix_encoding_mojibake(raw_description)
    url_s = str(url or "")
    combined = f"{title_s} | {desc_s[:3000]}"
    postal_code = ""
    cp_match = re.search(r"\b(?:C\.?P\.?\s*|ZIP\s*)?(20\d{3}|36\d{3}|37\d{3}|76\d{3}|98\d{3})\b", combined, re.I)
    if cp_match: postal_code = cp_match.group(1)

    municipality, state = "", ""
    norm_comb = normalize_text(f"{url_s} {title_s} {desc_s[:600]}")
    for mun_k, st_v in sorted(TARGET_MUNICIPALITIES.items(), key=lambda x: -len(x[0])):
        if re.search(rf"\b{mun_k}\b", norm_comb):
            municipality = mun_k.title()
            state = st_v
            break

    colonia = ""
    m_slug = re.search(r"/propiedad/[^/]+/[^/]+/[^/]+/([a-z0-9\-]+?)-id-[a-z0-9\-]+", url_s, re.I)
    if m_slug:
        slug_clean = m_slug.group(1).replace("-", " ").strip()
        if slug_clean and not slug_clean.isdigit(): colonia = slug_clean.title()

    if not colonia and title_s:
        t_clean = re.sub(r"\s*\([^)]*\)\s*-\s*icasas\.mx.*$", "", title_s, flags=re.I)
        t_clean = re.sub(r"^.*?Casa(?:\s+en\s+\w+)?\s+en\s+", "", t_clean, flags=re.I)
        parts = [p.strip() for p in t_clean.split(",") if p.strip()]
        stop_words = set(TARGET_MUNICIPALITIES.keys()) | {"MEXICO", "CIUDAD DE AGUASCALIENTES", "GUANAJUATO"}
        valid_parts = [p for p in parts if not re.match(r"^\d{5}$", p.strip()) and normalize_text(p) not in stop_words]
        if valid_parts: colonia = valid_parts[-1]

    if not colonia and desc_s:
        m_card = re.search(
            r"([A-Za-zÁÉÍÓÚÑáéíóúñ0-9\s\.]{3,45}),\s*(?:Aguascalientes|Jes[uú]s Mar[ií]a|"
            r"San Francisco de los Romo|Le[oó]n|Pur[ií]sima del Rinc[oó]n|San Francisco del Rinc[oó]n|"
            r"Quer[eé]taro|Corregidora|El Marqu[eé]s|Huimilpan|"
            r"Zacatecas|Guadalupe|Morelos|Vetagrande|Trancoso)\s*,\s*"
            r"(?:Aguascalientes|Guanajuato|Quer[eé]taro|Zacatecas)\s+\d+\s*rec", desc_s, re.I)
        if m_card: colonia = m_card.group(1).strip()

    if not colonia:
        m_col = re.search(r"\b(?:Colonia|Fraccionamiento|Privada|Residencial|Condominio)\s+([A-ZÁÉÍÓÚÑa-z0-9\s]{3,40}?)(?=\s*(?:,|-|\.|\n|\||C\.P|$))", combined, re.I)
        if m_col: colonia = m_col.group(1).strip(" ,.-|")

    return colonia, municipality, state, postal_code


def extract_generic_metadata(soup, url):
    data = {
        "title": "", "raw_description": "", "price": None, "currency": "MXN",
        "land_area_m2": None, "built_area_m2": None, "bedrooms": None,
        "bathrooms": None, "parking_spaces": None
    }
    og_title = soup.find("meta", property="og:title")
    if og_title and og_title.get("content"): data["title"] = fix_encoding_mojibake(og_title.get("content")).strip()
    elif soup.title: data["title"] = fix_encoding_mojibake(soup.title.string).strip()

    og_desc = soup.find("meta", property="og:description")
    if og_desc and og_desc.get("content"): data["raw_description"] = fix_encoding_mojibake(og_desc.get("content")).strip()

    for script in soup.find_all("script", type=re.compile(r"application/ld\+json", re.I)):
        try:
            parsed = json.loads(script.string or script.get_text())
            _walk_jsonld(parsed, data)
        except Exception: pass

    body_text = fix_encoding_mojibake(soup.get_text(" ", strip=True))
    if len(body_text) > len(data["raw_description"]):
        data["raw_description"] = f"{data['raw_description']} | {body_text[:3500]}".strip(" |")

    if data["price"] is None:
        p_val, c_val = parse_price_from_text(body_text)
        data["price"] = p_val
        if c_val: data["currency"] = c_val

    return data

def _walk_jsonld(node, data):
    if isinstance(node, list):
        for item in node: _walk_jsonld(item, data)
        return
    if not isinstance(node, dict): return
    if "@graph" in node: _walk_jsonld(node["@graph"], data)

    node_type = str(node.get("@type", "")).lower()
    if not data["title"] and node.get("name") and "website" not in node_type:
        data["title"] = fix_encoding_mojibake(node.get("name")).strip()
    if node.get("description") and len(str(node.get("description"))) > len(data["raw_description"]):
        data["raw_description"] = fix_encoding_mojibake(node.get("description")).strip()

    if data["price"] is None:
        raw_p = node.get("price")
        raw_c = node.get("priceCurrency")
        if raw_p is None and isinstance(node.get("offers"), dict):
            raw_p = node["offers"].get("price") or node["offers"].get("lowPrice")
            raw_c = node["offers"].get("priceCurrency")
        if raw_p is not None:
            p_val, c_val = parse_price_from_text(str(raw_p), json_price=raw_p, json_currency=raw_c)
            if p_val: data["price"] = p_val
            if c_val: data["currency"] = c_val

    for k, v in node.items():
        if k != "@graph" and isinstance(v, (dict, list)): _walk_jsonld(v, data)


def extract_cards_and_links_from_search_page(soup, page_url, source):
    detail_urls, direct_records = [], []
    containers = soup.find_all(["article", "li", "div"], class_=re.compile(r"listing-card|property-card|search-item|card|snippet|ad-data", re.I))
    
    for c in containers:
        txt = fix_encoding_mojibake(c.get_text(" ", strip=True))
        if len(txt) < 50: continue
        price, currency = parse_price_from_text(txt)
        if not price: continue

        heading = c.find(["h2", "h3", "a"])
        title = fix_encoding_mojibake(heading.get_text(" ", strip=True)) if heading else txt[:120]

        link_tag = c.find("a", href=True)
        card_link = urljoin(page_url, link_tag["href"]).split("#")[0] if link_tag else f"{page_url}#{hashlib.md5(title.encode()).hexdigest()[:8]}"

        col_txt, mun_txt, state_txt, cp_txt = extract_location_smart(title, txt, card_link)
        if not mun_txt: continue

        land_area, built_area = extract_areas_from_text(txt)
        beds, baths, parking = extract_rooms_and_parking_from_text(txt)
        built_area, beds, baths = apply_aggressive_fallback(f"{txt} {title}", built_area, beds, baths)
        is_remate = detect_remate_bancario(title, txt)

        direct_records.append({
            "source": source, "url_original": card_link, "title": title[:250], "operation": "sale",
            "property_type": "house", "price": price, "currency": currency or "MXN",
            "land_area_m2": land_area or "", "built_area_m2": built_area or "",
            "bedrooms": beds or "", "bathrooms": baths or "", "parking_spaces": parking or "",
            "street": "", "colonia_reported": col_txt, "municipality_reported": mun_txt,
            "state_reported": state_txt, "postal_code": cp_txt, "is_remate_bancario": is_remate,
            "raw_description": txt[:2500], "parser_status": "success", "notes": f"Extracted from {source} card"
        })

    for a in soup.find_all("a", href=True):
        h = a["href"].strip()
        if source == "icasas" and "/propiedad/" in h:
            detail_urls.append(urljoin(page_url, h).split("#")[0])

    next_page_url = None
    next_tag = soup.find(["link", "a"], rel="next") or soup.find("a", class_=re.compile(r"next|siguiente|pag-next", re.I))
    if next_tag and next_tag.get("href"):
        next_page_url = urljoin(page_url, next_tag["href"])

    return detail_urls, direct_records, next_page_url


def build_fallback_next_page_url(source, start_url, next_page_num):
    if source == "icasas": return f"{start_url.rstrip('/')}/p_{next_page_num}"
    parsed = urlparse(start_url)
    qs = parse_qs(parsed.query)
    qs["page"] = [str(next_page_num)]
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, parsed.params, urlencode(qs, doseq=True), ""))

def fetch_page(session, url):
    try:
        time.sleep(random.uniform(MIN_DELAY_SECONDS, MAX_DELAY_SECONDS))
        resp = session.get(url, headers={"User-Agent": USER_AGENT}, timeout=REQUEST_TIMEOUT)
        if resp.status_code in (401, 403, 429): return "blocked", None
        return "ok", resp
    except Exception: return "failed", None


def main():
    os.makedirs(INPUTS_DIR, exist_ok=True)
    print("=" * 78)
    print("STAGE 1: WEB SCRAPING & CONAPO COLONIA MATCHING (FLAGSHIP 4-CITY MATRIX)")
    print("=" * 78)
    
    gdf_conapo = load_conapo_shapefile(SHP_PATH)
    gdf_by_mun = {m: gdf_conapo[gdf_conapo["CVE_MUN_FULL"] == m].copy() for m in ALL_TARGET_MUN_CODES}

    session = requests.Session()
    unique_listing_queue = []
    all_direct_records = []
    seen_urls = set()

    for source, urls in SEARCH_URLS.items():
        for start_url in urls:
            curr_url = start_url
            for page in range(1, MAX_PAGES_PER_SOURCE + 1):
                if not curr_url or curr_url in seen_urls: break
                print(f"[{source}] Fetching P{page}: {curr_url}")
                seen_urls.add(curr_url)

                status, resp = fetch_page(session, curr_url)
                if status != "ok": break

                soup = BeautifulSoup(resp.text, "lxml")
                detail_urls, direct_recs, next_url = extract_cards_and_links_from_search_page(soup, curr_url, source)

                for r in direct_recs:
                    if r["url_original"] not in seen_urls:
                        seen_urls.add(r["url_original"])
                        all_direct_records.append(r)

                for u in detail_urls:
                    if u not in seen_urls:
                        unique_listing_queue.append((source, u))

                if not next_url and len(direct_recs) >= 5:
                    next_url = build_fallback_next_page_url(source, start_url, page + 1)
                curr_url = next_url

    print(f"\n[INFO] Discovery done. Found {len(all_direct_records)} search cards and {len(unique_listing_queue)} detail URLs.")

    final_records = []
    for r in all_direct_records:
        r["listing_id"] = f"{r['source'][:4].upper()}_{hashlib.md5(r['url_original'].encode()).hexdigest()[:8]}"
        r["scraped_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if not r.get("title") and r.get("colonia_reported"):
            r["title"] = f"Casa en Venta en {r['colonia_reported']}, {r.get('municipality_reported', '')}"
        
        m_method, m_score, cvegeo, col_nom, mun_c, st_c, clasif, imn = match_colonia_scoped(
            r.get("colonia_reported"), r.get("municipality_reported"), gdf_by_mun
        )
        r.update({"colonia_match_method": m_method, "colonia_match_score": m_score, "colonia_cvegeo": cvegeo,
                  "colonia_nombre": col_nom, "municipality_code": mun_c, "state_code": st_c,
                  "conapo_clasif": clasif, "conapo_imn2020": imn})
        final_records.append(r)

    for idx, (source, url) in enumerate(unique_listing_queue, 1):
        if url in seen_urls: continue
        seen_urls.add(url)
        print(f"[{idx}/{len(unique_listing_queue)}] Visiting: {url}")
        status, resp = fetch_page(session, url)
        if status != "ok": continue

        soup = BeautifulSoup(resp.text, "lxml")
        meta = extract_generic_metadata(soup, url)

        col_txt, mun_txt, state_txt, cp_txt = extract_location_smart(meta["title"], meta["raw_description"], url)
        if not mun_txt: continue

        land, built = extract_areas_from_text(meta["raw_description"])
        beds, baths, park = extract_rooms_and_parking_from_text(meta["raw_description"])
        built, beds, baths = apply_aggressive_fallback(f"{meta['raw_description']} {meta['title']}", built, beds, baths)

        rec = {
            "listing_id": f"{source[:4].upper()}_{hashlib.md5(url.encode()).hexdigest()[:8]}",
            "source": source, "url_original": url, "scraped_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "title": meta["title"][:250], "operation": "sale", "property_type": "house",
            "price": meta["price"] or "", "currency": meta["currency"] or "MXN",
            "land_area_m2": meta["land_area_m2"] or land or "",
            "built_area_m2": meta["built_area_m2"] or built or "",
            "bedrooms": meta["bedrooms"] or beds or "",
            "bathrooms": meta["bathrooms"] or baths or "",
            "parking_spaces": meta["parking_spaces"] or park or "",
            "street": "", "colonia_reported": col_txt,
            "municipality_reported": mun_txt, "state_reported": state_txt,
            "postal_code": cp_txt,
            "is_remate_bancario": detect_remate_bancario(meta["title"], meta["raw_description"]),
            "raw_description": meta["raw_description"][:2500],
            "parser_status": "success", "notes": "From Detail Page"
        }
        
        m_method, m_score, cvegeo, col_nom, mun_c, st_c, clasif, imn = match_colonia_scoped(
            rec["colonia_reported"], rec["municipality_reported"], gdf_by_mun
        )
        rec.update({"colonia_match_method": m_method, "colonia_match_score": m_score, "colonia_cvegeo": cvegeo,
                  "colonia_nombre": col_nom, "municipality_code": mun_c, "state_code": st_c,
                  "conapo_clasif": clasif, "conapo_imn2020": imn})
        final_records.append(rec)

    out_df = pd.DataFrame(final_records, columns=OUTPUT_COLUMNS)
    out_df.drop_duplicates(subset=["url_original"], inplace=True)
    out_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    
    print(f"\n[SUCCESS] Saved {len(out_df)} listings to {OUTPUT_CSV}")
    print(f"Matched {len(out_df[out_df['colonia_match_method'] != 'unmatched'])} listings to CONAPO colonias.")

if __name__ == "__main__":
    main()
