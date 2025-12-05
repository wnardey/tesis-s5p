# src/metrics.py
# -*- coding: utf-8 -*-
"""
Métricas de evaluación del ranking espacial multigás.

Incluye:
- build_eval_df: une el ranking (scores por celda) con la "verdad terreno"
  (df_inspecciones) y construye df_eval.
- compute_hr_area_pai: curva HR–Área, PAI, precisión y FDR por fracción de área.
- compute_auc_pr: curva Precisión–Recall y AUC–PR.
- fdr_at_hr: FDR@HR*, es decir, FDR en el primer punto donde HR >= HR_target.
"""

from typing import Tuple, Optional, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, average_precision_score


# ----------------------------------------------------------------------
# 1. Construir df_eval (ranking + verdad terreno)
# ----------------------------------------------------------------------

def build_eval_df(
    df_rank: pd.DataFrame,
    df_inspecciones: pd.DataFrame,
    score_col: str = "S_max",
) -> pd.DataFrame:
    """
    Construye el DataFrame de evaluación df_eval a partir de:
      - df_rank: ranking espacial (scores por celda)
      - df_inspecciones: celdas con fuentes conocidas (label = 1)

    Parámetros
    ----------
    df_rank : DataFrame
        Debe contener al menos:
          - 'cell_id'
          - columna de score (score_col)
        Opcionalmente:
          - 'lon', 'lat', 'year', 'month', etc. (se preservan).

    df_inspecciones : DataFrame
        Debe contener:
          - 'cell_id'
          - 'label' (1 si hay fuente conocida, 0 o NaN de lo contrario)

    score_col : str
        Nombre de la columna de score en df_rank (p.ej. 'S_max' o 'S_mean').

    Devuelve
    --------
    df_eval : DataFrame
        Contiene:
          - columnas originales de df_rank
          - 'score'  : score_col renombrado (para usar en métricas)
          - 'label'  : 1 si la celda tiene fuente conocida, 0 si no
    """
    df_r = df_rank.copy()
    df_i = df_inspecciones.copy()

    # Aseguramos tipos consistentes
    if "cell_id" in df_r.columns:
        df_r["cell_id"] = pd.to_numeric(df_r["cell_id"], errors="coerce").astype("Int64")
    if "cell_id" in df_i.columns:
        df_i["cell_id"] = pd.to_numeric(df_i["cell_id"], errors="coerce").astype("Int64")

    # Renombramos la columna de score a 'score' para estandarizar
    if score_col not in df_r.columns:
        raise KeyError(f"'{score_col}' no existe en df_rank.")
    df_r = df_r.rename(columns={score_col: "score"})

    # Nos quedamos con label en df_inspecciones
    if "label" not in df_i.columns:
        raise KeyError("'label' no existe en df_inspecciones.")
    df_i = df_i[["cell_id", "label"]].copy()

    # Outer join: queremos todas las celdas del ranking
    df_eval = df_r.merge(df_i, on="cell_id", how="left")

    # Celdas sin inspección conocida -> label = 0
    df_eval["label"] = df_eval["label"].fillna(0).astype(int)

    # Ordenamos por score descendente (ranking)
    df_eval = df_eval.sort_values("score", ascending=False).reset_index(drop=True)

    # Rank explícito (1 = mayor score)
    df_eval["rank"] = np.arange(1, len(df_eval) + 1)

    return df_eval


# ----------------------------------------------------------------------
# 2. Curva HR–Área, PAI, precisión y FDR
# ----------------------------------------------------------------------

def compute_hr_area_pai(
    df_eval: pd.DataFrame,
    score_col: str = "score",
) -> pd.DataFrame:
    """
    Calcula la curva HR–Área y el índice PAI, junto con precisión y FDR
    acumulados al ir inspeccionando celdas desde el score más alto.

    Parámetros
    ----------
    df_eval : DataFrame
        Debe contener:
          - 'score' (o score_col)
          - 'label' (0/1)

    score_col : str
        Nombre de la columna de score a usar.

    Devuelve
    --------
    df_hr : DataFrame
        Columnas:
          - 'rank'       : posición en el ranking (1 = mejor score)
          - 'k'          : número de celdas inspeccionadas (1..N)
          - 'area_frac'  : k / N
          - 'TP'         : verdaderos positivos acumulados
          - 'FP'         : falsos positivos acumulados
          - 'HR'         : TP / total_positivos
          - 'PAI'        : HR / area_frac  (si area_frac > 0)
          - 'precision'  : TP / (TP + FP)
          - 'FDR'        : 1 - precisión
    """
    df = df_eval.copy()

    if score_col not in df.columns:
        raise KeyError(f"'{score_col}' no existe en df_eval.")
    if "label" not in df.columns:
        raise KeyError("'label' no existe en df_eval.")

    # Ordenar por score descendente
    df = df.sort_values(score_col, ascending=False).reset_index(drop=True)

    # Variables básicas
    N = len(df)
    if N == 0:
        raise ValueError("df_eval está vacío.")

    df["k"] = np.arange(1, N + 1)           # número de celdas inspeccionadas
    df["area_frac"] = df["k"] / N          # fracción de área inspeccionada

    # TP/FP acumulados
    df["TP"] = df["label"].cumsum()
    df["FP"] = df["k"] - df["TP"]

    total_pos = df["label"].sum()

    if total_pos > 0:
        df["HR"] = df["TP"] / total_pos
    else:
        df["HR"] = 0.0

    # PAI: HR / area_frac (si area_frac > 0 y hay al menos un positivo)
    def _pai(row):
        if row["area_frac"] <= 0 or total_pos == 0:
            return 0.0
        return row["HR"] / row["area_frac"]

    df["PAI"] = df.apply(_pai, axis=1)

    # Precisión acumulada
    df["precision"] = df["TP"] / (df["TP"] + df["FP"])
    df["precision"] = df["precision"].fillna(0.0)

    # FDR = 1 - precisión
    df["FDR"] = 1.0 - df["precision"]

    # Rank explícito
    df["rank"] = df["k"].astype(int)

    return df


# ----------------------------------------------------------------------
# 3. Curva Precisión–Recall y AUC–PR
# ----------------------------------------------------------------------

def compute_auc_pr(
    df_eval: pd.DataFrame,
    score_col: str = "score",
) -> Tuple[Optional[float], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Calcula la curva Precisión–Recall y el AUC–PR (Average Precision)
    para el score dado.

    Parámetros
    ----------
    df_eval : DataFrame
        Debe contener:
          - 'label' (0/1)
          - 'score' (o score_col)

    score_col : str
        Nombre de la columna de score.

    Devuelve
    --------
    auc_pr : float o np.nan
    precision : np.ndarray o None
    recall : np.ndarray o None
    """
    if "label" not in df_eval.columns:
        raise KeyError("'label' no existe en df_eval.")
    if score_col not in df_eval.columns:
        raise KeyError(f"'{score_col}' no existe en df_eval.")

    y_true = df_eval["label"].astype(int).values
    scores = df_eval[score_col].astype(float).values

    n_pos = y_true.sum()
    if n_pos == 0:
        # No hay positivos: PR no está definida
        return np.nan, None, None

    precision, recall, _ = precision_recall_curve(y_true, scores)
    auc_pr = float(average_precision_score(y_true, scores))

    return auc_pr, precision, recall


# ----------------------------------------------------------------------
# 4. FDR@HR_target
# ----------------------------------------------------------------------

def fdr_at_hr(
    df_hr: pd.DataFrame,
    hr_target: float = 0.8,
) -> Dict[str, Optional[float]]:
    """
    Calcula FDR@HR_target: el FDR en el primer punto de la curva HR–Área
    donde HR >= hr_target.

    Parámetros
    ----------
    df_hr : DataFrame
        Salida de compute_hr_area_pai, debe contener:
          - 'HR'
          - 'area_frac'
          - 'precision'
          - 'FDR'
          - 'rank'

    hr_target : float
        Hit-rate objetivo (por ejemplo 0.8 o 0.9).

    Devuelve
    --------
    dict con:
      - 'HR_target'
      - 'area_frac'
      - 'precision'
      - 'FDR'
      - 'rank'
    """
    required_cols = {"HR", "area_frac", "precision", "FDR", "rank"}
    missing = required_cols.difference(df_hr.columns)
    if missing:
        raise KeyError(f"Faltan columnas en df_hr: {missing}")

    df_ok = df_hr[df_hr["HR"] >= hr_target]

    if df_ok.empty:
        return {
            "HR_target": hr_target,
            "area_frac": np.nan,
            "precision": np.nan,
            "FDR": np.nan,
            "rank": None,
        }

    row = df_ok.iloc[0]

    return {
        "HR_target": hr_target,
        "area_frac": float(row["area_frac"]),
        "precision": float(row["precision"]),
        "FDR": float(row["FDR"]),
        "rank": int(row["rank"]),
    }
