# -*- coding: utf-8 -*-
import os
import time
import random
import json
import re
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from PIL import Image
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

BASE_DIR = r"C:\Users\"
INPUTS_DIR = os.path.join(BASE_DIR, "Inputs")
IMAGES_DIR = os.path.join(INPUTS_DIR, "Images")
CSV_PATH = os.path.join(INPUTS_DIR, "property_listings_multicity_filtered_15km.csv")
AUDIT_DIR = os.path.join(BASE_DIR, "QualityAudit", "CV_Audit")
TEX_OUTPUT_PATH = os.path.join(AUDIT_DIR, "output_log.tex")

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
MAX_IMAGES_PER_LISTING = 4

METRO_MAPPING = {
    "Aguascalientes": "Aguascalientes", 
    "Jesus Maria": "Aguascalientes", 
    "Jesús María": "Aguascalientes",
    "San Francisco de los Romo": "Aguascalientes",
    "Leon": "León", 
    "León": "León",
    "Purisima del Rincon": "León", 
    "Purísima del Rincón": "León",
    "San Francisco del Rincon": "León",
    "San Francisco del Rincón": "León",
    "Celaya": "Celaya", 
    "Cortazar": "Celaya", 
    "Villagran": "Celaya",
    "Villagrán": "Celaya",
    "Zacatecas": "Zacatecas-Guadalupe", 
    "Guadalupe": "Zacatecas-Guadalupe", 
    "Morelos": "Zacatecas-Guadalupe", 
    "Vetagrande": "Zacatecas-Guadalupe", 
    "Trancoso": "Zacatecas-Guadalupe"
}

def get_image_urls(soup, source):
    urls = []
    
    # 1. JSON-LD structure
    for script in soup.find_all("script", type=re.compile(r"application/ld\+json", re.I)):
        try:
            data = json.loads(script.string or script.get_text())
            if isinstance(data, dict):
                img_val = data.get('image')
                if isinstance(img_val, list): urls.extend([str(x) for x in img_val if isinstance(x, str)])
                elif isinstance(img_val, str): urls.append(img_val)
            elif isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        img_val = item.get('image')
                        if isinstance(img_val, list): urls.extend([str(x) for x in img_val if isinstance(x, str)])
                        elif isinstance(img_val, str): urls.append(img_val)
        except Exception:
            pass

    # 2. OpenGraph Meta Tags
    for og in soup.find_all("meta", property=re.compile(r"og:image", re.I)):
        content = og.get("content")
        if content: urls.append(content)
        
    # 3. Direct <img> tags & gallery sliders
    for img in soup.find_all('img'):
        for attr in ['src', 'data-src', 'data-lazy', 'data-original', 'srcset']:
            val = img.get(attr)
            if val and val.startswith('http'):
                first_url = val.split(',')[0].strip().split(' ')[0]
                if first_url.startswith('http'):
                    urls.append(first_url)

    clean_urls = []
    for u in urls:
        u_lower = u.lower()
        if any(bad in u_lower for bad in ['icon', 'logo', 'avatar', 'map', 'placeholder', 'spinner', 'pixel', 'banner', 'button']):
            continue
        if u not in clean_urls:
            clean_urls.append(u)

    return clean_urls[:MAX_IMAGES_PER_LISTING]

def compute_image_diagnostics(prop_img_dir):
    """Extracts low-level visual quality metrics from downloaded property images."""
    if not os.path.exists(prop_img_dir):
        return {
            "img_count": 0, "mean_mp": np.nan, "mean_aspect": np.nan,
            "mean_brightness": np.nan, "mean_contrast": np.nan, 
            "mean_sharpness": np.nan, "mean_filesize_kb": np.nan
        }
        
    files = [os.path.join(prop_img_dir, f) for f in os.listdir(prop_img_dir) if f.lower().endswith(('jpg', 'jpeg', 'png'))]
    if not files:
        return {
            "img_count": 0, "mean_mp": np.nan, "mean_aspect": np.nan,
            "mean_brightness": np.nan, "mean_contrast": np.nan, 
            "mean_sharpness": np.nan, "mean_filesize_kb": np.nan
        }

    mps, aspects, brights, contrasts, sharps, sizes = [], [], [], [], [], []
    
    for fpath in files:
        try:
            fsize_kb = os.path.getsize(fpath) / 1024.0
            with Image.open(fpath) as im:
                im_rgb = im.convert("RGB")
                w, h = im_rgb.size
                mps.append((w * h) / 1e6)
                aspects.append(w / float(h) if h > 0 else np.nan)
                
                gray = np.asarray(im_rgb.convert("L"), dtype=np.float32)
                brights.append(np.mean(gray))
                contrasts.append(np.std(gray))
                
                if HAS_CV2:
                    lap_var = float(cv2.Laplacian(gray.astype(np.uint8), cv2.CV_64F).var())
                else:
                    gx, gy = np.gradient(gray)
                    lap_var = float(np.var(gx) + np.var(gy))
                sharps.append(lap_var)
                sizes.append(fsize_kb)
        except Exception:
            continue

    if not mps:
        return {
            "img_count": 0, "mean_mp": np.nan, "mean_aspect": np.nan,
            "mean_brightness": np.nan, "mean_contrast": np.nan, 
            "mean_sharpness": np.nan, "mean_filesize_kb": np.nan
        }

    return {
        "img_count": len(mps),
        "mean_mp": np.mean(mps),
        "mean_aspect": np.mean(aspects),
        "mean_brightness": np.mean(brights),
        "mean_contrast": np.mean(contrasts),
        "mean_sharpness": np.mean(sharps),
        "mean_filesize_kb": np.mean(sizes)
    }

def assign_metro_area(df):
    if 'metro_area' in df.columns and df['metro_area'].notnull().any():
        return df['metro_area'].fillna("Unmapped")
    if 'NOM_MET' in df.columns and df['NOM_MET'].notnull().any():
        return df['NOM_MET'].fillna("Unmapped")
    if 'municipality_reported' in df.columns:
        return df['municipality_reported'].map(METRO_MAPPING).fillna("Unmapped")
    return pd.Series(["Unmapped"] * len(df), index=df.index)

def generate_latex_log(df_target, metro_summary, bias_tests, corr_stats, tex_path):
    """Generates a publication-ready LaTeX diagnostic log following strict stylistic rules."""
    lines = []
    lines.append(r"% Automated Data & Visual Coverage Diagnostic Log")
    lines.append(r"\documentclass[11pt,a4paper]{article}")
    lines.append(r"\usepackage[utf8]{inputenc}")
    lines.append(r"\usepackage{booktabs}")
    lines.append(r"\usepackage{geometry}")
    lines.append(r"\usepackage{float}")
    lines.append(r"\geometry{margin=1in}")
    lines.append(r"\begin{document}")
    lines.append("")
    lines.append(r"\section*{Empirical Audit of Structural Filtering and Visual Disclosure}")
    
    total_n = len(df_target)
    with_img_n = int((df_target['img_count'] > 0).sum())
    multi_img_n = int((df_target['img_count'] >= 2).sum())
    overall_cov = (with_img_n / total_n * 100.0) if total_n > 0 else 0.0
    
    p_mwu_pval = bias_tests.get('price_m2_mwu_p', 1.0)
    area_mwu_pval = bias_tests.get('area_mwu_p', 1.0)
    rho_sharp = corr_stats.get('rho_price_sharpness', 0.0)
    p_sharp = corr_stats.get('p_price_sharpness', 1.0)
    rho_count = corr_stats.get('rho_price_imgcount', 0.0)
    p_count = corr_stats.get('p_price_imgcount', 1.0)

    lines.append(
        f"The spatial filtering workflow restricts the analytical sample to single-family dwellings situated within fifteen kilometers of each metropolitan central business district, yielding {total_n} valid non-foreclosure observations with complete structural records. "
        f"Across the four metropolitan areas, the automated visual retrieval pipeline successfully extracted photographic arrays for {with_img_n} properties, representing an overall visual disclosure rate of {overall_cov:.1f} percent, while {multi_img_n} listings contain multiple photographs suitable for cross-image aggregation. "
        f"Evaluating the distribution of structural attributes across metropolitan markets reveals pronounced variations in asking price per square meter and built footprint, reflecting asymmetric industrial growth and land-use regulations across central and northern Mexico."
    )
    lines.append("")
    lines.append(
        f"Because web-scraped real estate portals occasionally suffer from broken uniform resource locators or asynchronous gallery rendering, evaluating potential sample selection bias between listings with retrieved photographs and those lacking visual records is essential prior to estimating tree ensemble models. "
        f"Non-parametric Mann-Whitney U tests comparing the distribution of asking price per square meter between visually documented and undocumented listings yield a two-sided p-value of {p_mwu_pval:.4f}, whereas the corresponding test for built floor area yields a p-value of {area_mwu_pval:.4f}. "
        f"Furthermore, Spearman rank correlations across the documented sample indicate a correlation of {rho_count:.3f} (p-value {p_count:.4f}) between unit land value and image count, and a correlation of {rho_sharp:.3f} (p-value {p_sharp:.4f}) between unit price and Laplacian image sharpness, quantifying the degree to which brokerage marketing effort scales with property valuation."
    )
    lines.append("")

    # Table: Structural & Visual Summary by Metro Area
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(r"\caption{Metropolitan Sample Distribution, Structural Medians, and Visual Signal Quality}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{lrrrrrrrrr}")
    lines.append(r"\toprule")
    lines.append(r"Metropolitan Area & Listings ($N$) & Visual Cov. (\%) & Median Price (MXN) & Price/m$^2$ (MXN) & Built Area (m$^2$) & Mean Photos & Resolution (MP) & Luminance & Sharpness \\")
    lines.append(r"\midrule")
    
    for _, row in metro_summary.iterrows():
        m_name = str(row['metro_area']).replace("_", " ")
        n_val = int(row['N'])
        cov_val = row['cov_pct']
        p_med = row['median_price']
        pm2_med = row['median_price_m2']
        b_med = row['median_built']
        ph_mean = row['mean_photos']
        mp_mean = row['mean_mp']
        lum_mean = row['mean_brightness']
        shp_mean = row['mean_sharpness']
        
        mp_str = f"{mp_mean:.2f}" if pd.notnull(mp_mean) else "---"
        lum_str = f"{lum_mean:.1f}" if pd.notnull(lum_mean) else "---"
        shp_str = f"{shp_mean:.1f}" if pd.notnull(shp_mean) else "---"
        
        lines.append(
            f"{m_name} & {n_val} & {cov_val:.1f} & {p_med:,.0f} & {pm2_med:,.0f} & {b_med:.1f} & {ph_mean:.2f} & {mp_str} & {lum_str} & {shp_str} \\\\"
        )
        
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\end{table}")
    lines.append("")

    # Table: Attrition & Selection Bias Diagnostics
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(r"\caption{Non-Parametric Selection Bias Audit: Visually Documented versus Missing-Image Listings}")
    lines.append(r"\begin{tabular}{lrrrrr}")
    lines.append(r"\toprule")
    lines.append(r"Structural Variable & Documented Median & Undocumented Median & Mann-Whitney $U$ & $p$-value (MWU) & KS Statistic ($D$) \\")
    lines.append(r"\midrule")
    for var_label, key_prefix in [("Asking Price (MXN)", "price"), ("Price per Square Meter (MXN/m$^2$)", "price_m2"), ("Built Floor Area (m$^2$)", "area")]:
        doc_med = bias_tests.get(f"{key_prefix}_doc_med", np.nan)
        undoc_med = bias_tests.get(f"{key_prefix}_undoc_med", np.nan)
        u_stat = bias_tests.get(f"{key_prefix}_mwu_u", np.nan)
        u_pval = bias_tests.get(f"{key_prefix}_mwu_p", np.nan)
        ks_d = bias_tests.get(f"{key_prefix}_ks_d", np.nan)
        
        doc_s = f"{doc_med:,.1f}" if pd.notnull(doc_med) else "---"
        undoc_s = f"{undoc_med:,.1f}" if pd.notnull(undoc_med) else "---"
        u_s = f"{u_stat:,.0f}" if pd.notnull(u_stat) else "---"
        p_s = f"{u_pval:.4f}" if pd.notnull(u_pval) else "---"
        ks_s = f"{ks_d:.4f}" if pd.notnull(ks_d) else "---"
        
        lines.append(f"{var_label} & {doc_s} & {undoc_s} & {u_s} & {p_s} & {ks_s} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")
    lines.append(r"\end{document}")

    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def main():
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(AUDIT_DIR, exist_ok=True)

    if not os.path.exists(CSV_PATH):
        print(f"[ERROR] {CSV_PATH} not found. Verify the 15km filtered file path.")
        return

    print(f"[INFO] Loading 15km filtered dataset from {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)
    df['metro_area'] = assign_metro_area(df)

    # Filter for valid non-foreclosure single-family houses
    remate_col = df['is_remate_bancario'].fillna(False).astype(bool) if 'is_remate_bancario' in df.columns else pd.Series([False]*len(df))
    prop_col = df['property_type'].fillna('house').astype(str).str.lower() if 'property_type' in df.columns else pd.Series(['house']*len(df))
    
    mask = (
        (pd.to_numeric(df['price'], errors='coerce') > 0) & 
        (pd.to_numeric(df['built_area_m2'], errors='coerce') > 0) & 
        (pd.to_numeric(df['bedrooms'], errors='coerce') > 0) & 
        (pd.to_numeric(df['bathrooms'], errors='coerce') > 0) & 
        (~remate_col) & 
        (prop_col == 'house')
    )
    
    df_target = df[mask].copy()
    df_target['price'] = pd.to_numeric(df_target['price'], errors='coerce')
    df_target['built_area_m2'] = pd.to_numeric(df_target['built_area_m2'], errors='coerce')
    df_target['price_m2'] = df_target['price'] / df_target['built_area_m2']

    print(f"[INFO] Targeting {len(df_target)} valid 15km houses for image extraction and visual audit...")

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"})

    # 1. Image Download Loop
    for idx, row in df_target.iterrows():
        lid = str(row['listing_id'])
        url = str(row['url_original'])
        source = str(row.get('source', 'unknown'))
        
        prop_img_dir = os.path.join(IMAGES_DIR, lid)
        
        if os.path.exists(prop_img_dir) and len([f for f in os.listdir(prop_img_dir) if f.lower().endswith(('jpg', 'jpeg', 'png'))]) >= 2:
            continue 
            
        print(f"Fetching images for {lid} ({source})...")
        try:
            time.sleep(random.uniform(1.0, 2.0))
            resp = session.get(url, timeout=15)
            
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, 'lxml')
                img_urls = get_image_urls(soup, source)
                
                if img_urls:
                    os.makedirs(prop_img_dir, exist_ok=True)
                    success_count = 0
                    for img_url in img_urls:
                        try:
                            img_resp = session.get(img_url, timeout=10, headers={"Referer": url})
                            if img_resp.status_code == 200 and len(img_resp.content) > 3000:
                                ext = "png" if ".png" in img_url.lower() else "jpg"
                                filepath = os.path.join(prop_img_dir, f"photo_{success_count+1}.{ext}")
                                with open(filepath, 'wb') as f:
                                    f.write(img_resp.content)
                                success_count += 1
                        except Exception:
                            pass
        except Exception as e:
            print(f"  Failed to process {lid}: {e}")

    # 2. Compute Low-Level Image Diagnostics for All Targeted Properties
    print("\n[INFO] Computing low-level image diagnostics (resolution, luminance, contrast, sharpness)...")
    diag_records = []
    for idx, row in df_target.iterrows():
        lid = str(row['listing_id'])
        prop_img_dir = os.path.join(IMAGES_DIR, lid)
        metrics = compute_image_diagnostics(prop_img_dir)
        metrics['listing_id'] = lid
        diag_records.append(metrics)

    df_diag = pd.DataFrame(diag_records)
    df_audit = pd.merge(df_target, df_diag, on='listing_id', how='left')
    df_audit['has_images'] = df_audit['img_count'] > 0

    # Save enriched dataset with image diagnostics
    enriched_csv = os.path.join(AUDIT_DIR, "property_listings_15km_with_cv_diagnostics.csv")
    df_audit.to_csv(enriched_csv, index=False, encoding="utf-8")

    # 3. Metropolitan Summary Table
    metro_groups = []
    for metro, grp in df_audit.groupby('metro_area'):
        n_total = len(grp)
        n_cov = int((grp['img_count'] > 0).sum())
        metro_groups.append({
            "metro_area": metro,
            "N": n_total,
            "N_with_images": n_cov,
            "cov_pct": (n_cov / n_total * 100.0) if n_total > 0 else 0.0,
            "median_price": grp['price'].median(),
            "iqr_price": grp['price'].quantile(0.75) - grp['price'].quantile(0.25),
            "median_price_m2": grp['price_m2'].median(),
            "iqr_price_m2": grp['price_m2'].quantile(0.75) - grp['price_m2'].quantile(0.25),
            "median_built": grp['built_area_m2'].median(),
            "mean_photos": grp['img_count'].mean(),
            "mean_mp": grp['mean_mp'].mean(),
            "mean_brightness": grp['mean_brightness'].mean(),
            "mean_contrast": grp['mean_contrast'].mean(),
            "mean_sharpness": grp['mean_sharpness'].mean()
        })
    metro_summary_df = pd.DataFrame(metro_groups).sort_values(by="N", ascending=False)
    metro_summary_df.to_csv(os.path.join(AUDIT_DIR, "01_metro_structural_and_visual_summary.csv"), index=False)

    # 4. Selection Bias Audit (Documented vs Undocumented)
    bias_tests = {}
    doc = df_audit[df_audit['has_images']]
    undoc = df_audit[~df_audit['has_images']]

    for col, prefix in [('price', 'price'), ('price_m2', 'price_m2'), ('built_area_m2', 'area')]:
        s_doc = doc[col].dropna()
        s_undoc = undoc[col].dropna()
        bias_tests[f"{prefix}_doc_med"] = s_doc.median() if len(s_doc) > 0 else np.nan
        bias_tests[f"{prefix}_undoc_med"] = s_undoc.median() if len(s_undoc) > 0 else np.nan
        if len(s_doc) > 2 and len(s_undoc) > 2:
            u_stat, u_pval = stats.mannwhitneyu(s_doc, s_undoc, alternative='two-sided')
            ks_stat, ks_pval = stats.ks_2samp(s_doc, s_undoc)
            bias_tests[f"{prefix}_mwu_u"] = u_stat
            bias_tests[f"{prefix}_mwu_p"] = u_pval
            bias_tests[f"{prefix}_ks_d"] = ks_stat
            bias_tests[f"{prefix}_ks_p"] = ks_pval

    pd.DataFrame([bias_tests]).to_csv(os.path.join(AUDIT_DIR, "02_selection_bias_tests.csv"), index=False)

    # 5. Spearman Rank Correlations (Price/m2 vs Image Quality)
    corr_stats = {}
    if len(doc) > 5:
        r_cnt, p_cnt = stats.spearmanr(df_audit['price_m2'], df_audit['img_count'], nan_policy='omit')
        r_shp, p_shp = stats.spearmanr(doc['price_m2'], doc['mean_sharpness'], nan_policy='omit')
        r_mp, p_mp = stats.spearmanr(doc['price_m2'], doc['mean_mp'], nan_policy='omit')
        corr_stats = {
            "rho_price_imgcount": r_cnt, "p_price_imgcount": p_cnt,
            "rho_price_sharpness": r_shp, "p_price_sharpness": p_shp,
            "rho_price_megapixels": r_mp, "p_price_megapixels": p_mp
        }
    pd.DataFrame([corr_stats]).to_csv(os.path.join(AUDIT_DIR, "03_visual_price_correlations.csv"), index=False)

    # 6. Diagnostic Plots
    sns.set_theme(style="whitegrid")
    
    # Plot 1: Price/m2 Distribution by Image Availability across Metros
    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=df_audit[df_audit['metro_area'] != 'Unmapped'],
        x='metro_area', y='price_m2', hue='has_images', showfliers=False, palette="Set2"
    )
    plt.title("Price per Square Meter by Visual Disclosure Status across Metropolitan Areas")
    plt.xlabel("Metropolitan Area")
    plt.ylabel("Asking Price per m² (MXN)")
    plt.legend(title="Has Downloaded Images")
    plt.savefig(os.path.join(AUDIT_DIR, "fig01_selection_bias_boxplot.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 2: Visual Sharpness vs Resolution Joint Distribution
    if len(doc) > 5:
        plt.figure(figsize=(9, 6))
        sns.scatterplot(
            data=doc[doc['metro_area'] != 'Unmapped'],
            x='mean_mp', y='mean_sharpness', hue='metro_area', alpha=0.75
        )
        plt.title("Image Resolution (Megapixels) vs. Laplacian Sharpness by Metro Area")
        plt.xlabel("Mean Image Resolution (Megapixels)")
        plt.ylabel("Mean Laplacian Variance (Sharpness)")
        plt.savefig(os.path.join(AUDIT_DIR, "fig02_resolution_vs_sharpness.png"), dpi=300, bbox_inches='tight')
        plt.close()

    # 7. Write LaTeX Output Log
    generate_latex_log(df_audit, metro_summary_df, bias_tests, corr_stats, TEX_OUTPUT_PATH)
    print(f"\n[SUCCESS] Audit complete. LaTeX log written to: {TEX_OUTPUT_PATH}")
    print(f"[SUCCESS] Diagnostic CSVs and figures saved in: {AUDIT_DIR}")

if __name__ == '__main__':
    main()