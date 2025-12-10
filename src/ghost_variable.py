# src/ghost_variable.py
# -*- coding: utf-8 -*-
"""
Módulo para analizar la importancia de covariables mediante
el enfoque de *ghost variables*.

Idea básica:
- Para cada covariable X_j se ajusta un GAM univariado y se calcula
  un pseudo-R² (X_j "real").
- Luego se generan varias copias "fantasma" de X_j reordenando sus
  valores (rompiendo la relación con y) y se recalcula el pseudo-R².
- Si el pseudo-R² real es consistentemente mayor que la distribución
  de pseudo-R² de las variables fantasma, concluimos que X_j aporta
  señal real al modelo.

Funciones públicas:
- compute_ghost_importance(...)
- summarize_ghost_importance(...)
"""

from typing import List
import numpy as np
import pandas as pd
from pygam import LinearGAM, s


# ---------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------
def _safe_isnan(x) -> bool:
    """
    Devuelve True si x es NaN o si no se puede convertir de forma
    segura a float. Evita TypeError de np.isnan sobre tipos extraños.
    """
    try:
        return np.isnan(float(x))
    except Exception:
        return True


def _fit_univariate_gam_pseudo_r2(x: np.ndarray, y: np.ndarray) -> float:
    """
    Ajusta un GAM univariado y devuelve el pseudo-R² del modelo.

    Parameters
    ----------
    x : np.ndarray
        Vector columna (n, 1) con la covariable.
    y : np.ndarray
        Vector de respuesta (n,).

    Returns
    -------
    float
        Pseudo-R² del modelo ajustado. Si el ajuste falla, devuelve np.nan.
    """
    try:
        gam = LinearGAM(s(0))
        gam.fit(x, y)
        stats = getattr(gam, "statistics_", None) or {}
        return float(stats.get("pseudo_r2", np.nan))
    except Exception:
        return np.nan


# ---------------------------------------------------------------------
#  API principal
# ---------------------------------------------------------------------
def compute_ghost_importance(
    df: pd.DataFrame,
    pred_cols: List[str],
    response_col: str = "value",
    log_transform: bool = True,
    n_ghost: int = 20,
    random_state: int | None = None,
) -> pd.DataFrame:
    """
    Calcula la importancia tipo "ghost variable" para cada covariable
    listada en `pred_cols`.

    Para cada var en pred_cols:
      1) Ajusta un GAM univariado y obtiene pseudo_R² real.
      2) Genera n_ghost versiones aleatorias de esa var (barajando
         sus valores) y ajusta el GAM para cada una.
      3) Compara el pseudo_R² real con la distribución de ghosts.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con las covariables y la columna de respuesta.
    pred_cols : list of str
        Nombres de columnas de covariables a evaluar.
    response_col : str, default "value"
        Nombre de la columna de respuesta (gas).
    log_transform : bool, default True
        Si True, se trabaja con log(y) para estabilizar varianza.
    n_ghost : int, default 20
        Número de variables fantasma a generar por covariable.
    random_state : int or None, default None
        Semilla para el generador aleatorio.

    Returns
    -------
    pd.DataFrame
        DataFrame con una fila por covariable y columnas:

        - var
        - pseudo_r2_real
        - pseudo_r2_ghost_mean
        - pseudo_r2_ghost_std
        - delta_r2      (= real - mean(ghost))
        - p_ghost       (proporción de ghosts con R² ≥ real)
        - n_ghost       (número de ghosts válidos)
    """
    rng = np.random.default_rng(random_state)

    # --- Construir vector de respuesta ---
    y = df[response_col].to_numpy(dtype=float)
    if log_transform:
        y = np.log(np.clip(y, 1e-8, None))

    rows = []

    for var in pred_cols:
        # Extraer covariable como vector columna
        x = df[var].to_numpy(dtype=float).reshape(-1, 1)

        # 1) Pseudo-R² real con la covariable original
        r2_real = _fit_univariate_gam_pseudo_r2(x, y)

        # 2) Distribución de pseudo-R² para las ghost variables
        ghost_r2: list[float] = []

        for _ in range(n_ghost):
            try:
                x_ghost = x.copy()
                rng.shuffle(x_ghost)  # rompe relación x-y
                r2_g = _fit_univariate_gam_pseudo_r2(x_ghost, y)
                if not _safe_isnan(r2_g):
                    ghost_r2.append(float(r2_g))
            except Exception:
                # Si el ajuste falla en una iteración, la ignoramos
                continue

        # 3) Resumen estadístico
        if len(ghost_r2) == 0 or _safe_isnan(r2_real):
            # No hay base para comparar → marcamos como NaN
            rows.append(
                dict(
                    var=var,
                    pseudo_r2_real=(
                        float(r2_real) if not _safe_isnan(r2_real) else np.nan
                    ),
                    pseudo_r2_ghost_mean=np.nan,
                    pseudo_r2_ghost_std=np.nan,
                    delta_r2=np.nan,
                    p_ghost=np.nan,
                    n_ghost=n_ghost,
                )
            )
            continue

        ghost_arr = np.asarray(ghost_r2, dtype=float)
        ghost_mean = float(ghost_arr.mean())
        ghost_std = float(ghost_arr.std(ddof=1)) if ghost_arr.size > 1 else 0.0
        delta = float(r2_real - ghost_mean)
        # p_ghost: fracción de ghosts cuya pseudo_R² ≥ pseudo_R² real
        p_ghost = float((ghost_arr >= r2_real).mean())

        rows.append(
            dict(
                var=var,
                pseudo_r2_real=float(r2_real),
                pseudo_r2_ghost_mean=ghost_mean,
                pseudo_r2_ghost_std=ghost_std,
                delta_r2=delta,
                p_ghost=p_ghost,
                n_ghost=len(ghost_arr),
            )
        )

    return pd.DataFrame(rows)


def summarize_ghost_importance(df_imp: pd.DataFrame) -> pd.DataFrame:
    """
    Ordena y devuelve el resumen de importancia ghost.

    Parameters
    ----------
    df_imp : pd.DataFrame
        Salida de compute_ghost_importance.

    Returns
    -------
    pd.DataFrame
        Mismo DataFrame ordenado de mayor a menor delta_r2.
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
    # Mantener solo columnas esperadas si existen
    cols_present = [c for c in cols if c in df_imp.columns]

    return (
        df_imp[cols_present]
        .sort_values("delta_r2", ascending=False, na_position="last")
        .reset_index(drop=True)
    )
