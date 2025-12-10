# src/ghost_variable.py
# -*- coding: utf-8 -*-
"""
Utilidades para analizar importancia de covariables mediante
la metodología de 'ghost variables' usando GAM (pyGAM).

Enfoque:
- Ajustamos un GAM en la escala logarítmica de la variable de respuesta.
- Calculamos un pseudo-R² del modelo real.
- Para cada covariable, generamos n_ghost modelos donde esa columna
  es reemplazada por una 'ghost variable' (permuta aleatoria).
- Comparamos el pseudo-R² real vs. la distribución de pseudo-R² ghost.

Funciones públicas:
- compute_ghost_importance
- summarize_ghost_importance
"""

from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
from pygam import LinearGAM, s


# ---------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------

def _pseudo_r2(y_true: np.ndarray, y_hat: np.ndarray) -> float:
    """
    Pseudo-R² tipo R² clásico en la escala (log) de trabajo.

    R² = 1 - SSE/SST
    """
    y_true = np.asarray(y_true, dtype=float)
    y_hat = np.asarray(y_hat, dtype=float)

    mask = np.isfinite(y_true) & np.isfinite(y_hat)
    if mask.sum() < 3:
        return np.nan

    y_true = y_true[mask]
    y_hat = y_hat[mask]

    ss_res = np.sum((y_true - y_hat) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    if ss_tot <= 0:
        return np.nan

    return 1.0 - ss_res / ss_tot


def _build_gam(n_features: int, lam: Optional[np.ndarray] = None) -> LinearGAM:
    """
    Construye un LinearGAM con un término suave s(i) por feature.
    Si lam no es None, se fija ese valor de suavizado.
    """
    terms = None
    for i in range(n_features):
        terms = s(i) if terms is None else terms + s(i)

    gam = LinearGAM(terms)
    if lam is not None:
        gam.lam = lam
    return gam


# ---------------------------------------------------------------------
# API principal
# ---------------------------------------------------------------------

def compute_ghost_importance(
    df: pd.DataFrame,
    pred_cols: List[str],
    response_col: str = "value",
    log_transform: bool = True,
    n_ghost: int = 20,
    random_state: Optional[int] = None,
    min_n: int = 30,
) -> pd.DataFrame:
    """
    Calcula la importancia tipo 'ghost variable' para cada predictor.

    Parámetros
    ----------
    df : DataFrame
        Datos ya filtrados para un gas específico (una fila ~ celda-mes).
    pred_cols : list of str
        Nombres de las columnas predictoras (covariables meteorológicas).
    response_col : str
        Nombre de la columna respuesta (ej. 'value').
    log_transform : bool
        Si True, se trabaja en log(respuesta) para el GAM.
    n_ghost : int
        Número de réplicas ghost por variable.
    random_state : int o None
        Semilla para el generador aleatorio.
    min_n : int
        Mínimo de filas requeridas para correr el análisis.

    Devuelve
    --------
    DataFrame con columnas:
        ['var', 'pseudo_r2_real', 'pseudo_r2_ghost_mean',
         'pseudo_r2_ghost_std', 'delta_r2', 'p_ghost', 'n_ghost']
    """
    rng = np.random.default_rng(random_state)

    cols_needed = pred_cols + [response_col]
    df_work = df[cols_needed].dropna().copy()

    n = len(df_work)
    if n < min_n:
        # Demasiado pocos datos → devolvemos tabla vacía
        return pd.DataFrame(
            columns=[
                "var",
                "pseudo_r2_real",
                "pseudo_r2_ghost_mean",
                "pseudo_r2_ghost_std",
                "delta_r2",
                "p_ghost",
                "n_ghost",
            ]
        )

    # --------------------------------------------------------------
    # 1) Preparar X, y (con log-transform si aplica)
    # --------------------------------------------------------------
    X = df_work[pred_cols].to_numpy(dtype=float)
    y = df_work[response_col].to_numpy(dtype=float)

    if log_transform:
        y = np.log(np.clip(y, 1e-8, None))

    n_features = X.shape[1]

    # --------------------------------------------------------------
    # 2) Ajustar GAM completo (todas las covariables)
    # --------------------------------------------------------------
    gam_full = _build_gam(n_features)
    # gridsearch básico de lambda (podrías refinarlo si quieres)
    gam_full.gridsearch(X, y)

    y_hat_full = gam_full.predict(X)
    r2_full = _pseudo_r2(y, y_hat_full)

    rows: List[Dict[str, Any]] = []

    # --------------------------------------------------------------
    # 3) Para cada variable: reemplazarla por ghost y recomputar R²
    # --------------------------------------------------------------
    for j, var in enumerate(pred_cols):
        ghost_r2 = []

        for k in range(n_ghost):
            Xg = X.copy()

            # Ghost variable: permutación aleatoria de la columna j
            ghost_col = rng.permutation(X[:, j])
            Xg[:, j] = ghost_col

            gam_g = _build_gam(n_features, lam=gam_full.lam)
            try:
                gam_g.fit(Xg, y)
                y_hat_g = gam_g.predict(Xg)
                r2_g = _pseudo_r2(y, y_hat_g)
            except Exception:
                r2_g = np.nan

            if np.isfinite(r2_g):
                ghost_r2.append(r2_g)

        if len(ghost_r2) == 0 or not np.isfinite(r2_full):
            pseudo_r2_real = np.nan
            ghost_mean = np.nan
            ghost_std = np.nan
            delta_r2 = np.nan
            p_ghost = np.nan
        else:
            ghost_arr = np.asarray(ghost_r2, dtype=float)
            pseudo_r2_real = r2_full
            ghost_mean = float(ghost_arr.mean())
            ghost_std = float(ghost_arr.std(ddof=1)) if len(ghost_arr) > 1 else 0.0
            delta_r2 = max(0.0, r2_full - ghost_mean)
            # p_ghost: fracción de réplicas ghost cuyo pseudo-R² >= real
            p_ghost = float((ghost_arr >= r2_full).mean())

        rows.append(
            dict(
                var=var,
                pseudo_r2_real=pseudo_r2_real,
                pseudo_r2_ghost_mean=ghost_mean,
                pseudo_r2_ghost_std=ghost_std,
                delta_r2=delta_r2,
                p_ghost=p_ghost,
                n_ghost=len(ghost_r2),
            )
        )

    return pd.DataFrame(rows)


def summarize_ghost_importance(df_imp: pd.DataFrame) -> pd.DataFrame:
    """
    Ordena y deja solo las columnas clave del análisis ghost.
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

    if df_imp is None or df_imp.empty:
        return pd.DataFrame(columns=cols)

    df_out = df_imp.copy()
    for c in cols:
        if c not in df_out.columns:
            df_out[c] = np.nan

    df_out = df_out[cols]
    return df_out.sort_values("delta_r2", ascending=False).reset_index(drop=True)
