# src/ghost_variable.py
# -*- coding: utf-8 -*-
"""
Cálculo de importancia tipo "ghost variable" para modelos GAM.

Idea:
- Ajustar un GAM con todas las covariables.
- Medir un pseudo-R² (1 - RSS/TSS).
- Para cada covariable j:
    - Barajar (permutar) esa columna j n_ghost veces.
    - Reajustar el GAM y calcular el pseudo-R² 'fantasma'.
    - Comparar el pseudo-R² real vs distribución fantasma.
"""

from typing import List
import numpy as np
import pandas as pd
from pygam import LinearGAM, s


# -------- Helpers internos --------

def _build_terms(n_feat: int):
    """
    Construye s(0) + s(1) + ... + s(n_feat-1) correctamente,
    sin pasar por sum() que mete un 0 al inicio.
    """
    if n_feat <= 0:
        raise ValueError("n_feat debe ser >= 1")
    terms = s(0)
    for j in range(1, n_feat):
        terms += s(j)
    return terms


def _fit_gam_and_r2(X: np.ndarray, y: np.ndarray) -> float:
    """
    Ajusta un GAM lineal con un término suave por cada columna de X
    y devuelve un pseudo-R² = 1 - RSS/TSS.

    Devuelve NaN si no hay suficiente varianza o filas.
    """
    n_rows, n_feat = X.shape

    # Muy pocas filas -> no es confiable
    if n_rows < 10:
        return np.nan

    # Si y es casi constante, TSS ~ 0 -> R² no tiene sentido
    if np.allclose(y, y.mean()):
        return np.nan

    # Términos suaves s(0) + s(1) + ... + s(p-1)
    terms = _build_terms(n_feat)

    gam = LinearGAM(terms)
    gam.fit(X, y)

    y_hat = gam.predict(X)
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))

    if ss_tot <= 0:
        return np.nan

    r2 = 1.0 - ss_res / ss_tot
    return float(r2)


# -------- Función principal --------

def compute_ghost_importance(
    df: pd.DataFrame,
    pred_cols: List[str],
    response_col: str = "value",
    log_transform: bool = True,
    n_ghost: int = 20,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Calcula importancia de covariables vía ghost variables para un gas dado.

    df          : DataFrame filtrado a UN gas (e.g., solo NO2).
    pred_cols   : lista de nombres de columnas predictoras.
    response_col: nombre de la columna respuesta (valor del gas).
    log_transform: si True, se trabaja en log(y).
    n_ghost     : número de permutaciones por covariable.
    random_state: semilla para reproducibilidad.
    """

    rng = np.random.default_rng(random_state)

    cols_needed = pred_cols + [response_col]
    sub = df[cols_needed].copy()

    # Limpiar infinitos/NaN
    sub = sub.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")

    n_rows = sub.shape[0]
    if n_rows < 10:
        # Devuelve filas con NaN, pero estructura completa
        return pd.DataFrame({
            "var": pred_cols,
            "pseudo_r2_real": np.nan,
            "pseudo_r2_ghost_mean": np.nan,
            "pseudo_r2_ghost_std": np.nan,
            "delta_r2": np.nan,
            "p_ghost": np.nan,
            "n_ghost": n_ghost,
        })

    # Matrices numéricas
    X = sub[pred_cols].to_numpy(dtype=float)
    y = sub[response_col].to_numpy(dtype=float)

    if log_transform:
        y = np.log(np.clip(y, 1e-12, None))

    # R² base con todas las covariables
    base_r2 = _fit_gam_and_r2(X, y)

    rows = []

    for j, var in enumerate(pred_cols):
        ghost_r2_list = []

        for _ in range(n_ghost):
            X_perm = X.copy()
            # permutamos SOLO la columna j
            X_perm[:, j] = rng.permutation(X_perm[:, j])

            r2_g = _fit_gam_and_r2(X_perm, y)
            if not np.isnan(r2_g):
                ghost_r2_list.append(r2_g)

        if len(ghost_r2_list) == 0 or np.isnan(base_r2):
            rows.append({
                "var": var,
                "pseudo_r2_real": np.nan,
                "pseudo_r2_ghost_mean": np.nan,
                "pseudo_r2_ghost_std": np.nan,
                "delta_r2": np.nan,
                "p_ghost": np.nan,
                "n_ghost": n_ghost,
            })
            continue

        ghost_r2 = np.array(ghost_r2_list)
        ghost_mean = float(ghost_r2.mean())
        ghost_std = float(ghost_r2.std(ddof=1)) if len(ghost_r2) > 1 else 0.0

        # Mejora de R² frente a fantasmas (acotada a >= 0)
        delta_r2 = float(max(base_r2 - ghost_mean, 0.0))

        # p_ghost ~ probabilidad de que un modelo fantasma supere al real
        p_ghost = float((ghost_r2 >= base_r2).mean())

        rows.append({
            "var": var,
            "pseudo_r2_real": base_r2,
            "pseudo_r2_ghost_mean": ghost_mean,
            "pseudo_r2_ghost_std": ghost_std,
            "delta_r2": delta_r2,
            "p_ghost": p_ghost,
            "n_ghost": n_ghost,
        })

    return pd.DataFrame(rows)


def summarize_ghost_importance(df_imp: pd.DataFrame) -> pd.DataFrame:
    """
    Ordena por delta_r2 (de mayor a menor) y devuelve el resumen.
    """
    if df_imp is None or df_imp.empty:
        return df_imp

    cols = [
        "var",
        "pseudo_r2_real",
        "pseudo_r2_ghost_mean",
        "pseudo_r2_ghost_std",
        "delta_r2",
        "p_ghost",
        "n_ghost",
    ]
    df_out = df_imp.copy()
    df_out = df_out[cols]
    if "delta_r2" in df_out.columns:
        df_out = df_out.sort_values("delta_r2", ascending=False)

    return df_out.reset_index(drop=True)
