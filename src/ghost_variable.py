# src/ghost_variable.py
# -*- coding: utf-8 -*-
"""
Utilidades para análisis de importancia de covariables usando
el enfoque de 'ghost variables' sobre GAMs.

- compute_ghost_importance: calcula pseudo-R² real vs ghost por variable.
- summarize_ghost_importance: ordena y deja tabla lista para reporte.
- plot_ghost_heatmap: heatmap multigás de importancia relativa.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pygam import LinearGAM, s


# ---------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------


def _fit_gam(X: np.ndarray, y: np.ndarray) -> LinearGAM:
    """
    Ajusta un GAM lineal con un término suave por cada columna de X.
    Devuelve el modelo ya ajustado.
    """
    n_features = X.shape[1]
    terms = sum([s(i) for i in range(n_features)])

    gam = LinearGAM(terms)
    gam.gridsearch(X, y)
    return gam


def _pseudo_r2_from_gam(gam: LinearGAM) -> float:
    """
    Extrae el pseudo-R² desde el objeto GAM (pyGAM).
    Si no está disponible, devuelve NaN.
    """
    stats = getattr(gam, "statistics_", None)
    if stats is None:
        return np.nan
    return float(stats.get("pseudo_r2", np.nan))


# ---------------------------------------------------------------------
# API pública
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
    Calcula importancia tipo 'ghost variable' para cada covariable en pred_cols.

    Para cada variable v en pred_cols:
      1) Ajusta un GAM con todas las covariables reales -> pseudo_R2_real.
      2) Genera n_ghost "ghosts" para v (barajando la columna de v),
         re-ajusta el GAM cada vez -> pseudo_R2_ghost_i.
      3) Resume: media, std y p_ghost = P(R2_ghost >= R2_real).

    Devuelve un DataFrame con columnas:
      ['var', 'pseudo_r2_real', 'pseudo_r2_ghost_mean',
       'pseudo_r2_ghost_std', 'delta_r2', 'p_ghost', 'n_ghost']
    """

    rng = np.random.default_rng(random_state)

    # Respuesta
    y = df[response_col].to_numpy(dtype=float)
    if log_transform:
        y = np.log(np.clip(y, 1e-8, None))

    rows = []

    # GAM base: todas las covariables reales
    X_full = df[pred_cols].to_numpy(dtype=float)

    try:
        gam_real = _fit_gam(X_full, y)
        r2_real = _pseudo_r2_from_gam(gam_real)
    except Exception:
        # Si el ajuste base falla, no hay nada que hacer
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

    for var in pred_cols:
        ghost_r2: List[float] = []

        x_real = df[pred_cols].to_numpy(dtype=float)
        col_idx = pred_cols.index(var)
        original_col = x_real[:, col_idx].copy()

        for _ in range(n_ghost):
            x_ghost = x_real.copy()
            # Ghost = barajar la columna (misma distribución, rompe relación con y)
            shuffled = original_col.copy()
            rng.shuffle(shuffled)
            x_ghost[:, col_idx] = shuffled

            try:
                gam_g = _fit_gam(x_ghost, y)
                r2_g = _pseudo_r2_from_gam(gam_g)
                if np.isfinite(r2_g):
                    ghost_r2.append(r2_g)
            except Exception:
                # Si falla un ajuste ghost, se ignora
                continue

        if len(ghost_r2) == 0 or not np.isfinite(r2_real):
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

        ghost_r2 = np.array(ghost_r2, dtype=float)
        ghost_mean = float(ghost_r2.mean())
        ghost_std = float(ghost_r2.std(ddof=1)) if len(ghost_r2) > 1 else 0.0

        delta_r2 = max(r2_real - ghost_mean, 0.0)
        p_ghost = float((ghost_r2 >= r2_real).mean())

        rows.append(
            dict(
                var=var,
                pseudo_r2_real=float(r2_real),
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
    Ordena la tabla de importancia por delta_r2 (descendente) y
    devuelve un DataFrame listo para reporte.
    """
    if df_imp is None or df_imp.empty:
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

    cols = [
        "var",
        "pseudo_r2_real",
        "pseudo_r2_ghost_mean",
        "pseudo_r2_ghost_std",
        "delta_r2",
        "p_ghost",
        "n_ghost",
    ]
    existing_cols = [c for c in cols if c in df_imp.columns]

    df_out = df_imp[existing_cols].copy()
    df_out = df_out.sort_values("delta_r2", ascending=False)
    return df_out.reset_index(drop=True)


def plot_ghost_heatmap(
    ghost_results: Dict[str, pd.DataFrame],
    gases_order: List[str] | None = None,
    pred_cols_order: List[str] | None = None,
    figsize: tuple = (8, 6),
) -> None:
    """
    Construye un heatmap multigás con la importancia relativa de cada
    covariable (delta_r2 normalizado por gas).

    ghost_results: dict gas -> summary_df (salida de summarize_ghost_importance)
    gases_order: orden explícito de columnas (gases)
    pred_cols_order: orden explícito de filas (covariables)
    """

    if not ghost_results:
        print("⚠ ghost_results está vacío, nada que graficar.")
        return

    # Eje X: gases
    if gases_order is None:
        gases = list(ghost_results.keys())
    else:
        gases = [g for g in gases_order if g in ghost_results]

    # Eje Y: covariables
    if pred_cols_order is None:
        vars_all = sorted(
            {v for g in gases for v in ghost_results[g]["var"].tolist()}
        )
    else:
        vars_all = pred_cols_order

    n_vars = len(vars_all)
    n_gases = len(gases)

    # Matriz de delta_r2 normalizado
    mat = np.zeros((n_vars, n_gases), dtype=float)

    for j, gas in enumerate(gases):
        df_sum = ghost_results[gas].set_index("var")
        # vector de delta_r2 en el orden de vars_all
        delta_vec = np.array(
            [df_sum.loc[v, "delta_r2"] if v in df_sum.index else 0.0 for v in vars_all],
            dtype=float,
        )

        max_val = np.nanmax(delta_vec) if np.any(np.isfinite(delta_vec)) else 0.0
        if max_val > 0:
            delta_norm = delta_vec / max_val
        else:
            delta_norm = np.zeros_like(delta_vec)

        mat[:, j] = delta_norm

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(mat, aspect="auto", origin="lower", cmap="viridis")

    ax.set_xticks(np.arange(n_gases))
    ax.set_xticklabels(gases)
    ax.set_yticks(np.arange(n_vars))
    ax.set_yticklabels(vars_all)

    ax.set_xlabel("Gas")
    ax.set_ylabel("Covariable meteorológica")
    ax.set_title("Importancia relativa de covariables por gas\n"
                 "(Δ pseudo-R² normalizado)")

    # Anotar valores numéricos
    for i in range(n_vars):
        for j in range(n_gases):
            val = mat[i, j]
            ax.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                color="white" if val > 0.5 else "black",
                fontsize=8,
            )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Importancia relativa (0–1)")

    plt.tight_layout()
    plt.show()
