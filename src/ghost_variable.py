# src/ghost_variable.py
# -*- coding: utf-8 -*-
"""
Evaluación de importancia de variables mediante ghost variables usando GAM.

Idea básica:
- Para cada predictor X_j, se ajusta un GAM univariado y se calcula un pseudo-R²
  (tipo R² clásico) en la muestra.
- Se generan n_ghost réplicas "fantasma" barajando los valores de X_j
  (rompiendo la relación con la respuesta) y se ajusta el mismo GAM.
- Se compara el pseudo-R² de la variable real vs la distribución de pseudo-R²
  de las versiones ghost.

Salida:
- Un DataFrame con, por predictor:
    * pseudo_r2_real
    * pseudo_r2_ghost_mean
    * pseudo_r2_ghost_std
    * delta_r2 = real - media_ghost
    * p_ghost = fracción de ghosts que igualan o superan al real
"""

from typing import List

import numpy as np
import pandas as pd
from pygam import LinearGAM, s


def _fit_univariate_gam(X: np.ndarray, y: np.ndarray) -> float:
    """
    Ajusta un GAM univariado (s(0)) y devuelve un pseudo-R² tipo R² clásico:

        R² = 1 - sum((y - y_hat)^2) / sum((y - mean(y))^2)

    Ignora filas con NaN / inf.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    # Máscara de valores finitos
    mask = np.isfinite(X[:, 0]) & np.isfinite(y)
    Xm = X[mask]
    ym = y[mask]

    # Si hay muy pocos datos no tiene sentido ajustar
    if Xm.shape[0] < 5:
        return np.nan

    # Ajustar GAM univariado
    gam = LinearGAM(s(0))
    gam.gridsearch(Xm, ym)

    # Predicciones y pseudo-R²
    y_hat = gam.predict(Xm)
    ss_res = float(np.sum((ym - y_hat) ** 2))
    ss_tot = float(np.sum((ym - ym.mean()) ** 2))

    if ss_tot <= 0:
        return np.nan

    pseudo_r2 = 1.0 - ss_res / ss_tot
    return float(pseudo_r2)


def compute_ghost_importance(
    df: pd.DataFrame,
    pred_cols: List[str],
    response_col: str,
    log_transform: bool = True,
    min_value: float = 1e-8,
    n_ghost: int = 20,
    random_state: int = 0,
) -> pd.DataFrame:
    """
    Calcula importancia tipo ghost variable para cada predictor en pred_cols.

    df : DataFrame
        Debe contener la columna de respuesta (response_col) y los predictores
        en pred_cols. Cada fila es una observación (p.ej. celda–mes para un gas).
    pred_cols : list of str
        Nombres de las columnas predictoras (p.ej. ['T2m_C', 'BLH', ...]).
    response_col : str
        Nombre de la columna de respuesta (p.ej. 'value' del gas).
    log_transform : bool
        Si True, se aplica log a la respuesta: y = log(max(response_col, min_value)).
        Si False, se usa la respuesta tal cual.
    min_value : float
        Valor mínimo para recortar antes del log (evitar log(0)).
    n_ghost : int
        Número de réplicas ghost (barajadas) por predictor.
    random_state : int
        Semilla aleatoria para reproducibilidad.

    Devuelve
    --------
    df_imp : DataFrame
        Una fila por predictor, con columnas:
          - 'var'
          - 'pseudo_r2_real'
          - 'pseudo_r2_ghost_mean'
          - 'pseudo_r2_ghost_std'
          - 'delta_r2'
          - 'p_ghost'
          - 'n_ghost'
    """
    rng = np.random.default_rng(random_state)

    # --- Preparar respuesta y ---
    y_raw = df[response_col].astype(float).values
    if log_transform:
        y = np.log(np.clip(y_raw, min_value, None))
    else:
        y = y_raw

    results = []

    for var in pred_cols:
        if var not in df.columns:
            raise KeyError(f"El predictor '{var}' no existe en el DataFrame.")

        x_raw = df[var].astype(float).values.reshape(-1, 1)

        # 1) GAM real
        try:
            pseudo_r2_real = _fit_univariate_gam(x_raw, y)
        except Exception as e:
            print(f"[WARN] Error al ajustar GAM real para '{var}': {e}")
            pseudo_r2_real = np.nan

        # 2) GAMs ghost (barajando el predictor)
        ghost_r2 = []
        for _ in range(n_ghost):
            idx_perm = rng.permutation(len(x_raw))
            x_ghost = x_raw[idx_perm, :]

            try:
                r2_g = _fit_univariate_gam(x_ghost, y)
            except Exception as e:
                print(f"[WARN] Error al ajustar GAM ghost para '{var}': {e}")
                r2_g = np.nan

            ghost_r2.append(r2_g)

        ghost_r2 = np.array(ghost_r2, dtype=float)

        # Estadísticas ghost
        ghost_mean = float(np.nanmean(ghost_r2))
        ghost_std = float(np.nanstd(ghost_r2))

        # Diferencia de pseudo-R²
        delta_r2 = pseudo_r2_real - ghost_mean

        # p_ghost: fracción de ghosts que igualan o superan al real
        if np.isnan(pseudo_r2_real):
            p_ghost = np.nan
        else:
            p_ghost = float(np.mean(ghost_r2 >= pseudo_r2_real))

        results.append({
            "var": var,
            "pseudo_r2_real": pseudo_r2_real,
            "pseudo_r2_ghost_mean": ghost_mean,
            "pseudo_r2_ghost_std": ghost_std,
            "delta_r2": delta_r2,
            "p_ghost": p_ghost,
            "n_ghost": n_ghost,
        })

    df_imp = pd.DataFrame(results)
    df_imp = df_imp.sort_values("delta_r2", ascending=False).reset_index(drop=True)

    return df_imp


def summarize_ghost_importance(df_imp: pd.DataFrame) -> pd.DataFrame:
    """
    Utilidad simple para filtrar y ordenar resultados de ghost variable.

    Parámetros
    ----------
    df_imp : DataFrame
        Salida de compute_ghost_importance.

    Devuelve
    --------
    df_summary : DataFrame
        Mismas columnas, ordenadas por delta_r2 descendente.
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
    cols = [c for c in cols if c in df_imp.columns]
    df_summary = df_imp[cols].sort_values("delta_r2", ascending=False).reset_index(drop=True)
    return df_summary
