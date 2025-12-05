# src/metrics.py
# -*- coding: utf-8 -*-
"""
Métricas de priorización para el pipeline S5P/ERA5.

Incluye:
- build_eval_df: une ranking espacial con ground truth de inspecciones.
- compute_hr_area_pai: curva HR–Área y PAI.
- compute_auc_pr: AUC–PR (Average Precision) del ranking.
"""

from typing import Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, average_precision_score


def build_eval_df(
    df_rank: pd.DataFrame,
    df_inspecciones: pd.DataFrame,
    score_col: str = "S_max",
) -> pd.DataFrame:
    """
    Construye df_eval uniendo:
      - df_rank: ranking espacial por celda (incluye score_col),
      - df_inspecciones: ground truth con columnas ['cell_id', 'label'].

    Devuelve un DataFrame con:
      ['cell_id', 'lon', 'lat', 'score', 'label'].
    """
    # Tomamos solo lo necesario del ranking
    cols_needed = ["cell_id", "lon", "lat", score_col]
    df_rank2 = df_rank[cols_needed].copy()
    df_rank2 = df_rank2.rename(columns={score_col: "score"})

    # Unimos con ground truth (left join)
    df_eval = df_rank2.merge(df_inspecciones, on="cell_id", how="left")

    # Celdas sin label explícito se consideran 0 (no hallazgo)
    df_eval["label"] = df_eval["label"].fillna(0).astype(int)

    return df_eval


def compute_hr_area_pai(
    df_eval: pd.DataFrame,
    score_col: str = "score",
) -> pd.DataFrame:
    """
    Calcula, a partir de df_eval con columnas ['score', 'label']:

      - area_frac: fracción de área inspeccionada (asumiendo celdas de igual área),
      - HR: hit-rate acumulado (proporción de hallazgos detectados),
      - PAI: HR / area_frac (eficiencia vs inspección aleatoria).

    Devuelve un DataFrame ordenado por score descendente con columnas:
      ['cell_id', score_col, 'label', 'rank', 'area_frac', 'cum_hits', 'HR', 'PAI'].
    """
    df = df_eval[["cell_id", score_col, "label"]].copy()
    df = df.sort_values(score_col, ascending=False).reset_index(drop=True)

    n = len(df)
    total_hits = df["label"].sum()

    df["rank"] = np.arange(1, n + 1, dtype=float)
    df["area_frac"] = df["rank"] / n

    if total_hits > 0:
        df["cum_hits"] = df["label"].cumsum()
        df["HR"] = df["cum_hits"] / total_hits
    else:
        df["cum_hits"] = 0.0
        df["HR"] = 0.0

    # PAI: cuántas veces mejoras respecto a inspección aleatoria
    # (donde HR ≈ area_frac).
    df["PAI"] = df["HR"] / df["area_frac"]

    return df


def compute_auc_pr(
    df_eval: pd.DataFrame,
    score_col: str = "score",
) -> Tuple[float, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Calcula la curva Precisión–Recall y el AUC–PR (Average Precision)
    para el ranking definido por 'score_col'.

    Devuelve:
      - auc_pr: float (np.nan si no hay positivos),
      - precision: array o None,
      - recall: array o None.
    """
    y_true = df_eval["label"].values
    y_score = df_eval[score_col].values

    if y_true.sum() == 0:
        # No hay positivos, AUC-PR no es interpretable
        return float("nan"), None, None

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    auc_pr = average_precision_score(y_true, y_score)

    return auc_pr, precision, recall
