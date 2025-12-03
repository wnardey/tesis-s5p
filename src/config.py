# src/config.py
# -*- coding: utf-8 -*-
"""
Configuración central del proyecto S5P/ERA5.
Aquí definimos los parámetros por defecto para un experimento de prueba.
"""

from typing import Dict, Any


def get_default_params() -> Dict[str, Any]:
    """
    Devuelve un diccionario con los parámetros por defecto
    para el experimento (ejemplo: Mumbai, India, 5 km, NO2/CH4).
    """
    params: Dict[str, Any] = {
        # --- AOI (GAUL) ---
        "country": "India",
        "adm1_names": ["Maharashtra"],  # Nivel administrativo 1
        "adm_level": 2,                 # Se usa GAUL level2
        "adm2_includes": ["mumbai"],    # substrings (insensitive) en ADM2_NAME
        "buffer_km": 0.0,               # buffer opcional sobre el AOI
        "center_zoom": 9,               # zoom por defecto para visualización

        # --- Tiempo ---
        "years": [2024],
        "months": [8],  # agosto 2024 solo para probar

        # --- Gases a extraer ---
        "gases": ["NO2", "CH4", "O3", "CO"],

        # --- QA mínimos por gas (S5P) ---
        "qa_min": {
            "NO2": 0.75,
            "CH4": 0.50,
            "CO":  0.50,
            "SO2": 0.50,
            "HCHO": 0.50,
            "O3":  0.50,
        },

        # --- Grilla (fishnet) ---
        "grid_cell_km": 5.0,    # lado de la celda (km)
        "crs_scale_m": 5000,    # escala de muestreo (m) para reduceRegions
    }

    return params
