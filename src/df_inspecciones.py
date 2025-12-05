# src/df_inspecciones.py
# -*- coding: utf-8 -*-
"""
Construcción de df_inspecciones a partir de inventarios externos.

Por ahora implementa:
- build_df_inspecciones_gppd: usa el Global Power Plant Database (GPPD)
  en Google Earth Engine para marcar celdas que contienen plantas de
  generación eléctrica (fuentes potenciales de NOx, CO, etc.).
"""

from typing import List, Optional

import ee
import pandas as pd


def build_df_inspecciones_gppd(
    aoi_geom: ee.Geometry,
    grid_fc: ee.FeatureCollection,
    fuel_keep: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Construye un DataFrame df_inspecciones usando el inventario global
    de plantas de energía (WRI/GPPD/power_plants) en Earth Engine.

    Parámetros
    ----------
    aoi_geom : ee.Geometry
        Geometría del área de estudio (AOI), p.ej. Mumbai.
    grid_fc : ee.FeatureCollection
        Grilla de celdas (fishnet) con propiedades:
        - 'cell_id64' : identificador entero de la celda
        - 'lon', 'lat' : coordenadas del centroide de la celda
    fuel_keep : lista de str, opcional
        Lista de tipos de combustible a conservar (campo 'fuel1' en GPPD).
        Si es None, se usa ['Coal', 'Oil', 'Gas', 'Biomass', 'Waste'].

    Devuelve
    --------
    df_inspecciones : pd.DataFrame
        Columnas:
        - cell_id : identificador de celda (int)
        - lon     : longitud del centroide de la celda
        - lat     : latitud del centroide de la celda
        - label   : 1 si la celda contiene al menos una planta GPPD filtrada
    """
    if fuel_keep is None:
        fuel_keep = ["Coal", "Oil", "Gas", "Biomass", "Waste"]

    # 1) Cargar inventario GPPD
    gppd = ee.FeatureCollection("WRI/GPPD/power_plants")

    # 2) Filtrar por tipos de combustible relevantes
    gppd_filt = gppd.filter(ee.Filter.inList("fuel1", fuel_keep))

    # 3) Recortar al AOI
    gppd_aoi = gppd_filt.filterBounds(aoi_geom)

    # 4) Join espacial planta -> celda de la grilla
    join = ee.Join.inner()
    cond = ee.Filter.intersects(leftField=".geo", rightField=".geo")

    joined = join.apply(primary=gppd_aoi, secondary=grid_fc, condition=cond)

    def map_join(fe):
        """
        Toma el resultado del join (primary=planta, secondary=celda)
        y devuelve una Feature con:
        - cell_id, cell_lon, cell_lat
        """
        plant = ee.Feature(fe.get("primary"))
        cell = ee.Feature(fe.get("secondary"))
        return plant.set({
            "cell_id": cell.get("cell_id64"),
            "cell_lon": cell.get("lon"),
            "cell_lat": cell.get("lat"),
        })

    plants_with_cells = joined.map(map_join)

    # 5) Traer al lado de Python las celdas que tienen al menos una planta
    #    (AOI es pequeño, así que el número de features es manejable)
    features = plants_with_cells.getInfo().get("features", [])

    rows = []
    for f in features:
        props = f.get("properties", {})
        cell_id = props.get("cell_id")
        lon = props.get("cell_lon")
        lat = props.get("cell_lat")
        if cell_id is not None:
            rows.append({
                "cell_id": int(cell_id),
                "lon": lon,
                "lat": lat,
            })

    if not rows:
        # No hay plantas en el AOI
        df_inspecciones = pd.DataFrame(columns=["cell_id", "lon", "lat", "label"])
        return df_inspecciones

    df_inspecciones = pd.DataFrame(rows)

    # Nos quedamos con una fila por celda
    df_inspecciones = (
        df_inspecciones
        .drop_duplicates(subset=["cell_id"])
        .reset_index(drop=True)
    )

    # Añadimos label = 1 (celda con fuente conocida)
    df_inspecciones["label"] = 1

    return df_inspecciones
