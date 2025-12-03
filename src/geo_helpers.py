# src/geo_helpers.py
# -*- coding: utf-8 -*-
"""
Helpers geoespaciales:
- Construcción de AOI a partir de GAUL (FAO/GAUL 2015, niveles 1/2).
- Construcción de grilla cuadrada (fishnet) sobre el AOI.
"""

from typing import Tuple, List
import math

import ee


def _gaul_collection(level: int) -> ee.FeatureCollection:
    """
    Devuelve la FeatureCollection de GAUL según el nivel.
    level = 1 -> nivel ADM1
    level = 2 -> nivel ADM2
    """
    if level == 1:
        return ee.FeatureCollection("FAO/GAUL/2015/level1")
    elif level == 2:
        return ee.FeatureCollection("FAO/GAUL/2015/level2")
    else:
        raise ValueError("adm_level debe ser 1 o 2 (GAUL).")


def build_aoi(params: dict) -> Tuple[ee.Geometry, ee.FeatureCollection]:
    """
    Construye el AOI (geometry + FeatureCollection) a partir de:
      - country
      - adm1_names
      - adm_level
      - adm2_includes
      - buffer_km

    Además imprime un pequeño resumen (área, número de elementos, nombres ADM).
    """
    country: str = params["country"]
    adm1_names: List[str] = params.get("adm1_names", [])
    level: int = int(params.get("adm_level", 2))
    includes = [s.lower() for s in params.get("adm2_includes", [])]
    buffer_km: float = float(params.get("buffer_km", 0.0))

    gaul = _gaul_collection(level)

    if level == 1:
        # Filtrado solo hasta nivel ADM1
        fc = gaul.filter(ee.Filter.eq("ADM0_NAME", country))
        if adm1_names:
            fc = fc.filter(ee.Filter.inList("ADM1_NAME", adm1_names))
        sel = fc
    else:
        # Nivel 2: primero filtramos país y ADM1, luego ADM2 por substrings
        base = gaul.filter(ee.Filter.eq("ADM0_NAME", country))
        if adm1_names:
            base = base.filter(ee.Filter.inList("ADM1_NAME", adm1_names))

        if includes:
            fc_list = [
                base.filter(ee.Filter.stringContains("ADM2_NAME", s.capitalize()))
                for s in includes
            ]
            sel = fc_list[0]
            for f in fc_list[1:]:
                sel = sel.merge(f)
        else:
            sel = base

    # Geometría disuelta de todo el conjunto seleccionado
    geom = sel.geometry().dissolve()

    # Buffer opcional
    if buffer_km != 0:
        geom = geom.buffer(buffer_km * 1000)

    # FeatureCollection con una sola feature (el AOI)
    aoi_fc = ee.FeatureCollection([ee.Feature(geom)])

    # Resumen para consola
    try:
        name_field = "ADM2_NAME" if level == 2 else "ADM1_NAME"
        names = sel.aggregate_array(name_field).getInfo()
    except Exception:
        names = []

    try:
        area_km2 = geom.area().divide(1e6).getInfo()
    except Exception:
        area_km2 = None

    print("\n🌍 Resumen AOI (GAUL):")
    print(f"  País           : {country}")
    print(f"  Nivel GAUL     : {level}")
    print(f"  ADM1           : {adm1_names}")
    print(f"  Filtro ADM2    : {params.get('adm2_includes', [])}")
    if area_km2 is not None:
        print(f"  Área total AOI : {area_km2:,.1f} km²")
    if names:
        unique_names = sorted(set(names))
        print("  Nombres ADM    : " + ", ".join(unique_names[:10]) +
              (" ..." if len(unique_names) > 10 else ""))

    return geom, aoi_fc


def build_square_grid(
    geom: ee.Geometry,
    cell_km: float,
    scale_m: float,
) -> ee.FeatureCollection:
    """
    Construye una grilla cuadrada (fishnet) sobre el AOI.

    Cada celda:
      - Tiene un identificador entero 'cell_id64'
      - Se le añaden atributos 'lon' y 'lat' del centroide
    """
    cell_m = cell_km * 1000.0

    # Latitud del centro para corrección aproximada de distancia este-oeste
    centroid = geom.centroid(maxError=1)
    centroid_lat = centroid.coordinates().get(1).getInfo()

    # Imagen de lat/lon en grados
    ll = ee.Image.pixelLonLat()
    lon = ll.select("longitude")
    lat = ll.select("latitude")

    # Transformación a una rejilla (índices x, y) en metros aproximados
    x = (
        lon
        .multiply(math.cos(math.radians(centroid_lat)))
        .multiply(111000)
        .divide(cell_m)
        .floor()
    )
    y = (
        lat
        .multiply(111000)
        .divide(cell_m)
        .floor()
    )

    # ID único por celda (combinación de x e y)
    cell_id = x.multiply(1e6).add(y).toInt64().rename("cell_id64")
    grid_img = cell_id.clip(geom)

    # Convertimos la imagen de IDs en polígonos vectoriales
    vectors = grid_img.reduceToVectors(
        geometry=geom,
        scale=scale_m,
        geometryType="polygon",
        labelProperty="cell_id64",
        maxPixels=1e12,
    )

    # Añadimos centroides (lon, lat) como propiedades
    def add_centroid(fe):
        ctr = fe.geometry().centroid(1)
        lon_c = ctr.coordinates().get(0)
        lat_c = ctr.coordinates().get(1)
        return fe.set({"lon": lon_c, "lat": lat_c})

    grid_fc = vectors.map(add_centroid)

    print(f"\n🧩 Grilla construida: {grid_fc.size().getInfo()} celdas "
          f"(lado ≈ {cell_km} km)")
    return grid_fc
