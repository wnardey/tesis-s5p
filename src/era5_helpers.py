# src/era5_helpers.py
# -*- coding: utf-8 -*-
"""
Helpers para ERA5/ERA5-Land:
- era5_monthly: imagen mensual con T2m, precipitación, BLH, viento, etc.
- band_mean: media espacial de una banda sobre un AOI.
"""

from typing import Optional

import ee
import math

# IDs de colecciones ERA5
ERA5_ATM = "ECMWF/ERA5/HOURLY"
ERA5_LAND = "ECMWF/ERA5_LAND/HOURLY"

ERA5_ATM_BANDS = {
    "u10": "u_component_of_wind_10m",
    "v10": "v_component_of_wind_10m",
    "blh": "boundary_layer_height",
}
ERA5_LAND_BANDS = {
    "t2m": "temperature_2m",
    "tp": "total_precipitation",
}


def era5_monthly(start: ee.Date, end: ee.Date, geom: ee.Geometry) -> ee.Image:
    """
    Construye una imagen mensual promediada/sumada sobre el rango [start, end)
    para las variables ERA5/ERA5-Land más relevantes:

        - T2m_K      : temperatura a 2m (K)
        - precip     : precipitación acumulada (m)
        - BLH        : altura de la capa límite (m)
        - wind_speed : velocidad del viento a 10m (m/s)
        - wind_dir_deg: dirección del viento (grados 0–360)

    Devuelve una ee.Image con todas estas bandas recortadas al AOI.
    """
    # ERA5 atmósfera
    ic_atm = (
        ee.ImageCollection(ERA5_ATM)
        .filterDate(start, end)
        .filterBounds(geom)
        .select([
            ERA5_ATM_BANDS["u10"],
            ERA5_ATM_BANDS["v10"],
            ERA5_ATM_BANDS["blh"],
        ])
    )
    atm_mean = ic_atm.mean().clip(geom)

    # ERA5-Land
    ic_land = (
        ee.ImageCollection(ERA5_LAND)
        .filterDate(start, end)
        .filterBounds(geom)
        .select([ERA5_LAND_BANDS["t2m"], ERA5_LAND_BANDS["tp"]])
    )

    land_mean_t2m = ic_land.select(ERA5_LAND_BANDS["t2m"]).mean()
    land_sum_tp = ic_land.select(ERA5_LAND_BANDS["tp"]).sum()
    land_img = land_mean_t2m.addBands(land_sum_tp).clip(geom)

    # Unimos atm + land
    img = atm_mean.addBands(land_img)

    # Derivados de viento
    u = img.select(ERA5_ATM_BANDS["u10"])
    v = img.select(ERA5_ATM_BANDS["v10"])

    wind_speed = u.pow(2).add(v.pow(2)).sqrt().rename("wind_speed")
    wind_dir = (
        u.multiply(-1)
        .atan2(v.multiply(-1))
        .multiply(180 / math.pi)
        .mod(360)
        .rename("wind_dir_deg")
    )

    img = img.addBands([wind_speed, wind_dir]).rename([
        ERA5_ATM_BANDS["u10"],      # u10
        ERA5_ATM_BANDS["v10"],      # v10
        "BLH",                      # boundary_layer_height
        "T2m_K",                    # temperatura 2m (K)
        "precip",                   # precipitación (m)
        "wind_speed",               # velocidad viento (m/s)
        "wind_dir_deg",             # dirección viento (grados)
    ])

    return img


def band_mean(
    img: ee.Image,
    band_name: str,
    geom: ee.Geometry,
    scale_m: float,
) -> Optional[float]:
    """
    Calcula la media espacial de una banda de ERA5 sobre el AOI.
    Devuelve un float o None si falla.
    """
    img_b = img.select(band_name).rename("x")
    try:
        mean_dict = img_b.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geom,
            scale=scale_m,
            maxPixels=1e12,
        ).getInfo()
        return (mean_dict or {}).get("x", None)
    except Exception:
        return None
