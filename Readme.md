# Polycentricity, Gated Enclaves, and the Bundling of Visual Quality in Mexican Metropolitan Housing Markets


This repository contains the end-to-end Python pipeline, computer vision feature extraction workflow, spatial harmonization scripts, and anonymized multimodal dataset accompanying the research article:

> **Polycentricity, Gated Enclaves, and the Bundling of Visual Quality in Mexican Metropolitan Housing Markets**

## Overview

The codebase integrates three data modalities across four intermediate Mexican metropolitan areas (**Querétaro, Aguascalientes, León, and Zacatecas-Guadalupe**) within a 15-kilometer urban core radius ($N = 3,099$ properties across $572$ official *colonia* polygons):
1. **Structural & Morphological Attributes:** Web-scraped residential property dimensions harmonized with 9-digit official settlement keys (`CVEGEO`) from the National Population Council (CONAPO).
2. **Exogenous Polycentric Subcenters & Local Amenity Buffers:** Employment-weighted retail and industrial manufacturing subcenters and scale-invariant 1-km amenity/disamenity log-densities computed offline from 168,686 geocoded establishments in the National Statistical Directory of Economic Units (INEGI DENUE).
3. **Zero-Shot Computer Vision (`openai/clip-vit-base-patch32`) & Photometrics:** Sequential three-block vision-language classification masking non-photographic advertising noise, quantifying interior room disclosure, and scoring structural conservation, streetscape greenery, and interior finish modernity.

## Repository Structure

* `data/anonymized/`: Contains the clean, anonymized analytical dataset (`mexican_housing_multimodal_15km_anonymized.csv`) required to directly reproduce all econometric models, spatial cross-validation benchmarks, and figures.
* `data/external/`: Directory for placing raw CONAPO cartography (`colonias_imc2020.shp`) and INEGI DENUE microdata CSVs when running the upstream spatial extraction scripts from scratch.
* `scripts/`:
  * `01_scraping.py`: Multi-portal web scraper and municipality-scoped CONAPO *colonia* matcher.
  * `02_filter_15km_and_audit.py`: Plausibility filtering, foreclosure (*remates bancarios*) exclusion, and 15-km core boundary truncation.
  * `03_extract_images.py`: Property gallery image downloader and non-parametric visual selection bias audit.
  * `04_clip_vision_pipeline.ipynb`: Zero-shot CLIP (`openai/clip-vit-base-patch32`) and OpenCV photometric feature extraction pipeline.
  * `05_spatial_features_denue.py`: Offline INEGI DENUE spatial buffer builder and employment-weighted K-Means subcenter estimator.
  * `06_econometric_ml_pipeline.py`: Master estimation script running the 5-model progressive OLS sequence, within-*colonia* fixed effects, spatial-block `GroupKFold` cross-validation, TreeSHAP attributions, partial dependence profiles, and 300 DPI publication cartography.
* `outputs/`: Destination folder for generated LaTeX tables (`outputs/tables/`) and 300 DPI figures (`outputs/figures/`).

