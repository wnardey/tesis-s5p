# src/config.py
# -*- coding: utf-8 -*-
"""
Configuración central del proyecto S5P/ERA5.
Aquí definimos los parámetros por defecto para un experimento de prueba.
"""

from typing import Dict, Any


def get_default_params(
    years=None,
    months=None,
    gases=None,
) -> Dict[str, Any]:
    """
    Devuelve el diccionario de parámetros base del pipeline.

    Parámetros
    ----------
    years  : lista de años a procesar (ej. [2023, 2024]).
    months : lista de meses (1–12).
    gases  : lista de gases a extraer (claves usadas en el pipeline S5P).
             Ejemplo: ["NO2", "CH4", "O3_TCL", "CO"].
    """

    # Valores por defecto si no se pasan
    if years is None:
        years = [2024]
    if months is None:
        months = [8]
    if gases is None:
        gases = ["NO2", "CH4", "O3_TCL", "CO"]

    params: Dict[str, Any] = {
        # --- AOI (GAUL) ---
        "country": "India",
        "adm1_names": ["Maharashtra"],   # Nivel administrativo 1
        "adm_level": 2,                  # Se usa GAUL level2
        "adm2_includes": ["mumbai"],     # substrings (insensitive) en ADM2_NAME
        "buffer_km": 0.0,                # buffer opcional sobre el AOI
        "center_zoom": 9,                # zoom por defecto para visualización

        # --- Tiempo ---
        "years": years,
        "months": months,

        # --- Gases a extraer ---
        "gases": gases,

        # --- QA mínimos por gas (S5P) ---
        "qa_min": {
            "NO2": 0.75,
            "CH4": 0.50,
            "CO":  0.50,
            "SO2": 0.50,
            "HCHO": 0.50,
            "O3_TCL": 0.50,   # importante: misma clave que en s5p_helpers
        },

        # --- Grilla (fishnet) ---
        "grid_cell_km": 5.0,   # lado de la celda (km)
        "crs_scale_m": 5000,   # escala de muestreo (m) para reduceRegions
    }

    return params
