# src/ghost_variable.py
# -*- coding: utf-8 -*-
"""
Utilidad de *ghost variables* para evaluar la importancia de covariables
meteorológicas en un GAM univariado.

Ideas clave:
- Para cada covariable X_j se ajusta un GAM univariado y → pseudo-R²_real.
- Luego se crean n_ghost versiones "fantasma" de X_j barajadas y se vuelve
  a ajustar el modelo → distribución de pseudo-R²_ghost.
- La importancia de la covariable se resume como:
      ΔR² = R²_real - media(R²_ghost)
  y un p-valor empírico:
      p_ghost = frac( R²_ghost ≥ R²_real )
"""

from typing import List
import numpy as np
import pandas as pd
from pygam import LinearGAM, s


# ---------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------


def _pseudo_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Pseudo-R² simple tipo R² clásico sobre la escala de trabajo (normalmente log).

    R² = 1 - SSE/SST

    Si la varianza de y_true es ~0 o algo sale raro, devuelve np.nan.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    if ss_tot <= 0:
        return np.nan

    ss_res = np.sum((y_true - y_pred) ** 2)
    return 1.0 - ss_res / ss_tot


def _fit_univariate_gam(
    X: np.ndarray,
    y: np.ndarray,
    progress: bool = False,
) -> float:
    """
    Ajusta un GAM univariado y devuelve el pseudo-R².

    X: matriz (n, 1)
    y: vector (n,)

    Si algo falla, devuelve np.nan.
    """
    try:
        gam = LinearGAM(s(0)).gridsearch(X, y, progress=progress)
        y_hat = gam.predict(X)
        r2 = _pseudo_r2(y, y_hat)
        if not np.isfinite(r2):
            return np.nan
        return float(r2)
    except Exception:
        return np.nan


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
    min_n: int = 10,
) -> pd.DataFrame:
    """
    Calcula importancia tipo "ghost variables" para cada covariable en pred_cols.

    Parámetros
    ----------
    df : DataFrame
        Debe contener columnas pred_cols y response_col.
    pred_cols : list[str]
        Nombres de las covariables a evaluar (una por GAM univariado).
    response_col : str
        Nombre de la columna respuesta (ej. 'value' para el gas).
    log_transform : bool
        Si True, se trabaja con log(y) (útil para gases positivos).
    n_ghost : int
        Número de replicaciones fantasma por covariable.
    random_state : int o None
        Semilla para la baraja aleatoria de las variables fantasma.
    min_n : int
        Número mínimo de observaciones válidas para intentar ajustar un modelo.

    Devuelve
    --------
    df_imp : DataFrame con columnas:
        - var         : nombre de la covariable
        - kind        : 'real' o 'ghost'
        - ghost_id    : -1 para real, 0..n_ghost-1 para fantasmas
        - pseudo_r2   : pseudo-R² del modelo
        - n_samples   : número de muestras usadas en ese ajuste
    """
    rng = np.random.default_rng(random_state)
    rows = []

    for var in pred_cols:
        # Seleccionamos solo la covariable y la respuesta
        sub = df[[var, response_col]].copy()

        # Limpiamos NaN e infinitos
        sub = sub.replace([np.inf, -np.inf], np.nan).dropna()
        y = sub[response_col].astype(float).values
        X = sub[[var]].astype(float).values  # shape (n, 1)

        # Transformación log si aplica
        if log_transform:
            y = np.clip(y, 1e-12, None)
            y = np.log(y)

        n = len(y)

        # Si no hay suficientes datos o y es casi constante -> no se puede
        if n < min_n or np.allclose(y, y.mean()):
            # Registramos filas con NaN para real y fantasmas
            rows.append(
                dict(
                    var=var,
                    kind="real",
                    ghost_id=-1,
                    pseudo_r2=np.nan,
                    n_samples=n,
                )
            )
            for k in range(n_ghost):
                rows.append(
                    dict(
                        var=var,
                        kind="ghost",
                        ghost_id=k,
                        pseudo_r2=np.nan,
                        n_samples=n,
                    )
                )
            continue

        # --- Modelo real ---
        r2_real = _fit_univariate_gam(X, y, progress=False)
        rows.append(
            dict(
                var=var,
                kind="real",
                ghost_id=-1,
                pseudo_r2=r2_real,
                n_samples=n,
            )
        )

        # --- Modelos fantasma ---
        for k in range(n_ghost):
            Xg = X.copy()
            rng.shuffle(Xg[:, 0])  # barajamos la covariable

            r2_g = _fit_univariate_gam(Xg, y, progress=False)
            rows.append(
                dict(
                    var=var,
                    kind="ghost",
                    ghost_id=k,
                    pseudo_r2=r2_g,
                    n_samples=n,
                )
            )

    return pd.DataFrame(rows)


def summarize_ghost_importance(df_imp: pd.DataFrame) -> pd.DataFrame:
    """
    Resume el resultado de compute_ghost_importance a nivel de covariable.

    Parámetros
    ----------
    df_imp : DataFrame
        Salida de compute_ghost_importance.

    Devuelve
    --------
    summary : DataFrame con columnas:
        - var
        - pseudo_r2_real
        - pseudo_r2_ghost_mean
        - pseudo_r2_ghost_std
        - delta_r2          (real - media ghost)
        - p_ghost           (frac. de R²_ghost >= R²_real)
        - n_ghost           (nº de réplicas fantasma usadas)
    """
    rows = []

    for var, sub in df_imp.groupby("var"):
        real_vals = sub[sub["kind"] == "real"]["pseudo_r2"].values
        ghost_vals = sub[sub["kind"] == "ghost"]["pseudo_r2"].values

        r2_real = real_vals[0] if len(real_vals) > 0 else np.nan
        r2_real = float(r2_real) if np.isfinite(r2_real) else np.nan

        ghost_finite = ghost_vals[np.isfinite(ghost_vals)]

        if len(ghost_finite) == 0 or not np.isfinite(r2_real):
            rows.append(
                dict(
                    var=var,
                    pseudo_r2_real=np.nan,
                    pseudo_r2_ghost_mean=np.nan,
                    pseudo_r2_ghost_std=np.nan,
                    delta_r2=np.nan,
                    p_ghost=np.nan,
                    n_ghost=len(ghost_finite),
                )
            )
            continue

        mu = float(ghost_finite.mean())
        sd = float(ghost_finite.std(ddof=1)) if len(ghost_finite) > 1 else 0.0
        delta = float(r2_real - mu)

        # p_ghost: fracción de fantasmas con R² >= R²_real
        p = float(np.mean(ghost_finite >= r2_real))

        rows.append(
            dict(
                var=var,
                pseudo_r2_real=r2_real,
                pseudo_r2_ghost_mean=mu,
                pseudo_r2_ghost_std=sd,
                delta_r2=delta,
                p_ghost=p,
                n_ghost=len(ghost_finite),
            )
        )

    summary = pd.DataFrame(rows)
    if not summary.empty and "delta_r2" in summary.columns:
        summary = summary.sort_values("delta_r2", ascending=False).reset_index(drop=True)

    return summary
