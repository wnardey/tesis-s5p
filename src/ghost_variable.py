# src/ghost_variable.py
# -*- coding: utf-8 -*-
"""
Herramientas para evaluar la importancia de covariables usando
el enfoque de "ghost variables" con GAMs univariados.

Para cada covariable:
- Se ajusta un GAM univariado real (x_real -> y).
- Se ajustan n_ghost GAMs barajando la covariable (x_ghost -> y).
- Se compara el pseudo-R² real vs la distribución ghost.

Salidas clave:
- pseudo_r2_real
- pseudo_r2_ghost_mean
- pseudo_r2_ghost_std
- delta_r2 = real - ghost_mean
- p_ghost = P(R²_ghost >= R²_real)
"""

from typing import List, Dict

import numpy as np
import pandas as pd
from pygam import LinearGAM, s


def _fit_univariate_gam(x: np.ndarray, y: np.ndarray) -> float:
    """
    Ajusta un GAM univariado y devuelve el pseudo-R².
    Lanza excepción si algo va mal (se maneja fuera).
    """
    gam = LinearGAM(s(0), fit_intercept=True)
    gam.gridsearch(x, y)
    pseudo_r2 = gam.statistics_["pseudo_r2"]
    return pseudo_r2


def compute_ghost_importance(
    df: pd.DataFrame,
    pred_cols: List[str],
    response_col: str = "value",
    log_transform: bool = True,
    n_ghost: int = 20,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Calcula la importancia tipo ghost para cada covariable en pred_cols.

    Parámetros
    ----------
    df : DataFrame con las columnas pred_cols y response_col.
    pred_cols : lista de nombres de covariables.
    response_col : nombre de la columna respuesta (ej: 'value').
    log_transform : si True, se usa log(y) como respuesta (clipeando a >0).
    n_ghost : número de permutaciones ghost por covariable.
    random_state : semilla para reproducibilidad.

    Devuelve
    --------
    DataFrame con columnas:
      - var
      - pseudo_r2_real
      - pseudo_r2_ghost_mean
      - pseudo_r2_ghost_std
      - delta_r2
      - p_ghost
      - n_ghost
    """
    rng = np.random.default_rng(random_state)

    y = df[response_col].to_numpy(dtype=float)

    # Transformación log si se solicita
    if log_transform:
        y = np.log(np.clip(y, 1e-8, None))

    rows: List[Dict] = []

    for var in pred_cols:
        x = df[[var]].to_numpy(dtype=float)

        # Chequeos básicos
        if np.isnan(x).any() or np.isnan(y).any():
            rows.append(
                dict(
                    var=var,
                    pseudo_r2_real=np.nan,
                    pseudo_r2_ghost_mean=np.nan,
                    pseudo_r2_ghost_std=np.nan,
                    delta_r2=np.nan,
                    p_ghost=np.nan,
                    n_ghost=n_ghost,
                )
            )
            continue

        if np.std(x) == 0 or np.std(y) == 0:
            # Sin varianza no tiene sentido ajustar
            rows.append(
                dict(
                    var=var,
                    pseudo_r2_real=np.nan,
                    pseudo_r2_ghost_mean=np.nan,
                    pseudo_r2_ghost_std=np.nan,
                    delta_r2=np.nan,
                    p_ghost=np.nan,
                    n_ghost=n_ghost,
                )
            )
            continue

        # --- Modelo real ---
        try:
            r2_real = _fit_univariate_gam(x, y)
        except Exception:
            r2_real = np.nan

        ghost_r2 = []
        # --- Modelos ghost (permutando la covariable) ---
        for _ in range(n_ghost):
            x_perm = x.copy()
            rng.shuffle(x_perm)
            try:
                r2_g = _fit_univariate_gam(x_perm, y)
                ghost_r2.append(r2_g)
            except Exception:
                # Si falla uno, lo ignoramos
                continue

        if len(ghost_r2) == 0 or np.isnan(r2_real):
            rows.append(
                dict(
                    var=var,
                    pseudo_r2_real=r2_real,
                    pseudo_r2_ghost_mean=np.nan,
                    pseudo_r2_ghost_std=np.nan,
                    delta_r2=np.nan,
                    p_ghost=np.nan,
                    n_ghost=n_ghost,
                )
            )
            continue

        ghost_r2 = np.array(ghost_r2)
        ghost_mean = ghost_r2.mean()
        ghost_std = ghost_r2.std(ddof=1)

        delta_r2 = r2_real - ghost_mean

        # p_ghost: fracción de ghosts que igualan o superan el R² real
        p_ghost = float((ghost_r2 >= r2_real).mean())

        rows.append(
            dict(
                var=var,
                pseudo_r2_real=float(r2_real),
                pseudo_r2_ghost_mean=float(ghost_mean),
                pseudo_r2_ghost_std=float(ghost_std),
                delta_r2=float(delta_r2),
                p_ghost=p_ghost,
                n_ghost=n_ghost,
            )
        )

    return pd.DataFrame(rows)


def summarize_ghost_importance(df_imp: pd.DataFrame) -> pd.DataFrame:
    """
    Ordena el DataFrame de importancia por delta_r2 (descendente)
    y devuelve solo columnas clave en un orden legible.
    """
    cols = [
        "var",
        "pseudo_r2_real",
        "pseudo_r2_ghost_mean",
        "pseudo_r2_ghost_std",
        "delta_r2",
        "p_ghost",
        "n_ghost",
    ]
    out = df_imp[cols].copy()
    out = out.sort_values("delta_r2", ascending=False).reset_index(drop=True)
    return out
