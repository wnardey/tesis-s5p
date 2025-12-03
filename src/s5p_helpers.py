# src/s5p_helpers.py
# -*- coding: utf-8 -*-
"""
Helpers para Sentinel-5P (S5P):
- Definición de especificaciones por gas (colecciones, bandas, QA, unidades).
- Función para elegir automáticamente la mejor colección disponible
  en un periodo (start, end) sobre un AOI.
- Función para calcular estadísticas de QA (qa_frac) sobre el AOI.
- Función inventario_s5p(...) que devuelve un DataFrame con el inventario.
"""

from typing import Dict, Tuple, Optional
import ee
import pandas as pd

# --------------------------------------------------------------------
# Especificaciones de gases (colecciones L3 OFFL/NRTI + banda principal)
# --------------------------------------------------------------------
GAS_SPECS: Dict[str, Dict] = {
    "NO2": {
        "collections": [
            "COPERNICUS/S5P/OFFL/L3_NO2",
            "COPERNICUS/S5P/NRTI/L3_NO2",
        ],
        "band": "tropospheric_NO2_column_number_density",  # mol/m^2
        "qa": "qa_value",
        "units": "mol m^-2",
        "long_name": "Dióxido de nitrógeno troposférico",
    },
    "CH4": {
        "collections": [
            "COPERNICUS/S5P/OFFL/L3_CH4",
        ],
        "band": "CH4_column_volume_mixing_ratio_dry_air",
        "qa": "qa_value",
        "units": "ppb",
        "long_name": "Metano columna promediada (XCH4)",
    },
    "O3_TCL": {
        "collections": [
            "COPERNICUS/S5P/OFFL/L3_O3_TCL",
        ],
        "band": "ozone_tropospheric_vertical_column",  # mol/m^2
        "qa": "qa_value",
        "units": "mol m^-2",
        "long_name": "Ozono troposférico (columna troposférica)",
    },
    "CO": {
        "collections": [
            "COPERNICUS/S5P/OFFL/L3_CO",
            "COPERNICUS/S5P/NRTI/L3_CO",
        ],
        "band": "CO_column_number_density",  # mol/m^2
        "qa": "qa_value",
        "units": "mol m^-2",
        "long_name": "Monóxido de carbono (columna total)",
    },
}


# --------------------------------------------------------------------
# Selector de colección: elige la primera colección con datos (>0 imágenes)
# --------------------------------------------------------------------
def choose_ic(
    col_ids,
    start: ee.Date,
    end: ee.Date,
    geom: ee.Geometry,
) -> Tuple[Optional[str], Optional[ee.ImageCollection]]:
    """
    Dada una lista de IDs de colecciones S5P (L3) y un AOI + rango temporal,
    devuelve (collection_id, ImageCollection) para la primera que tenga
    al menos 1 imagen en el periodo/AOI.

    Si ninguna tiene datos, devuelve (None, None).
    """

    def _try_load(col_id: str) -> Optional[ee.ImageCollection]:
        ic = (
            ee.ImageCollection(col_id)
            .filterDate(start, end)
            .filterBounds(geom)
        )
        try:
            n = ic.size().getInfo()
            if n and n > 0:
                return ic
            return None
        except Exception:
            return None

    for cid in col_ids:
        ic = _try_load(cid)
        if ic is not None:
            return cid, ic

    return None, None


# --------------------------------------------------------------------
# Cálculo de estadísticas de QA (qa_frac) sobre el AOI
# --------------------------------------------------------------------
def compute_qa_stats(
    ic: ee.ImageCollection,
    gas_spec: Dict,
    qa_min: float,
    geom: ee.Geometry,
    scale_m: float,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Calcula estadísticas de la fracción de píxeles que pasan el umbral de QA
    (qa_value >= qa_min), a nivel de AOI:

      - qa_frac_mean: promedio espacial de qa_frac
      - qa_frac_min : mínimo espacial de qa_frac
      - qa_frac_max : máximo espacial de qa_frac
    """
    qa_band = gas_spec["qa"]

    def add_qa_mask(img):
        bands = img.bandNames()
        qa_img = ee.Image(ee.Algorithms.If(
            bands.contains(qa_band),
            img.select(qa_band).gte(qa_min).rename("qa_mask"),
            ee.Image.constant(1).rename("qa_mask"),
        ))
        return qa_img

    qa_ic = ic.map(add_qa_mask)

    qa_sum = qa_ic.select("qa_mask").sum().clip(geom)
    qa_cnt = qa_ic.select("qa_mask").count().clip(geom)
    qa_cnt_safe = qa_cnt.where(qa_cnt.eq(0), 1)
    qa_frac_img = qa_sum.divide(qa_cnt_safe).rename("qa_frac")

    try:
        mean_dict = qa_frac_img.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geom,
            scale=scale_m,
            maxPixels=1e12,
        ).getInfo()
        min_dict = qa_frac_img.reduceRegion(
            reducer=ee.Reducer.min(),
            geometry=geom,
            scale=scale_m,
            maxPixels=1e12,
        ).getInfo()
        max_dict = qa_frac_img.reduceRegion(
            reducer=ee.Reducer.max(),
            geometry=geom,
            scale=scale_m,
            maxPixels=1e12,
        ).getInfo()

        qa_mean = (mean_dict or {}).get("qa_frac", None)
        qa_min_sp = (min_dict or {}).get("qa_frac", None)
        qa_max_sp = (max_dict or {}).get("qa_frac", None)

        def _clamp01(x):
            if x is None:
                return None
            return max(0.0, min(1.0, float(x)))

        qa_mean = _clamp01(qa_mean)
        qa_min_sp = _clamp01(qa_min_sp)
        qa_max_sp = _clamp01(qa_max_sp)

    except Exception:
        return None, None, None

    return qa_mean, qa_min_sp, qa_max_sp


# --------------------------------------------------------------------
# Inventario S5P (ED_02): DataFrame por (año, mes, gas)
# --------------------------------------------------------------------
def inventario_s5p(params: Dict, aoi_geom: ee.Geometry) -> pd.DataFrame:
    """
    Construye un inventario tipo tabla (DataFrame) con:
        year, month, gas, gas_key, collection_id, n_images,
        qa_min, qa_frac_mean, qa_frac_min, qa_frac_max, units.
    """
    scale_m = params["crs_scale_m"]
    years = params["years"]
    months = params["months"]
    qa_min_cfg = params["qa_min"]

    records = []

    for year in years:
        for month in months:
            start = ee.Date.fromYMD(year, month, 1)
            if month == 12:
                end = ee.Date.fromYMD(year + 1, 1, 1)
            else:
                end = ee.Date.fromYMD(year, month + 1, 1)

            ym_label = f"{year}-{month:02d}"
            print(f"\n===== Inventario S5P para {ym_label} =====")

            for gas in params["gases"]:
                # Mapeo especial para ozono troposférico
                if gas == "O3":
                    gas_key = "O3_TCL"
                elif gas == "O3_TCL":
                    gas_key = "O3_TCL"
                else:
                    gas_key = gas

                if gas_key not in GAS_SPECS:
                    print(f"  [{gas}] ⚠ No hay especificación en GAS_SPECS, se omite.")
                    continue

                spec = GAS_SPECS[gas_key]
                qa_min_val = float(qa_min_cfg.get(gas, qa_min_cfg.get(gas_key, 0.5)))

                cid, ic = choose_ic(spec["collections"], start, end, aoi_geom)
                if cid is None or ic is None:
                    print(f"  [{gas}] ❌ Sin datos en colecciones candidatas.")
                    records.append({
                        "year": year,
                        "month": month,
                        "gas": gas,
                        "gas_key": gas_key,
                        "collection_id": None,
                        "n_images": 0,
                        "qa_min": qa_min_val,
                        "qa_frac_mean": None,
                        "qa_frac_min": None,
                        "qa_frac_max": None,
                        "units": spec["units"],
                    })
                    continue

                n_images = ic.size().getInfo()
                print(f"  [{gas}] Colección elegida: {cid} | n_imágenes = {n_images} | QA_min = {qa_min_val}")

                qa_mean, qa_min_sp, qa_max_sp = compute_qa_stats(
                    ic=ic,
                    gas_spec=spec,
                    qa_min=qa_min_val,
                    geom=aoi_geom,
                    scale_m=scale_m,
                )

                if qa_mean is not None:
                    print(
                        f"     → qa_frac_mean ≈ {qa_mean:.3f} | "
                        f"qa_frac_min ≈ {qa_min_sp:.3f} | "
                        f"qa_frac_max ≈ {qa_max_sp:.3f}"
                    )
                else:
                    print("     → qa_frac_* no disponible (error en reducción).")

                records.append({
                    "year": year,
                    "month": month,
                    "gas": gas,
                    "gas_key": gas_key,
                    "collection_id": cid,
                    "n_images": n_images,
                    "qa_min": qa_min_val,
                    "qa_frac_mean": qa_mean,
                    "qa_frac_min": qa_min_sp,
                    "qa_frac_max": qa_max_sp,
                    "units": spec["units"],
                })

    inv_s5p = pd.DataFrame(records)
    return inv_s5p
