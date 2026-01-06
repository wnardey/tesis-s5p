# src/concurvity.py
# -*- coding: utf-8 -*-
"""
Utilidades para evaluar concurvity en modelos GAM (pygam).

Idea:
- Si la versión de pygam expone `gam.statistics_['concurv']`, usamos directamente
  esa salida como medida de concurvity por término suave.
- Si no está disponible, calculamos una aproximación:
  para cada término suave f_j(X_j), vemos cuánto puede ser explicado por la
  combinación de los otros términos suaves f_-j(X_-j), vía una regresión lineal.
  El R² de esa regresión se interpreta como "qué tanto f_j está contenido
  en el espacio generado por el resto de términos" (concurvity alta = más redundancia).
"""

from typing import List, Optional

import numpy as np
import pandas as pd
from pygam import LinearGAM, GAM


def _safe_get_statistics(gam: GAM):
    """
    Intenta obtener el diccionario de estadísticas del GAM.
    Algunas versiones usan `statistics_`, otras exponen `statistics()`.
    Si ninguna funciona, devuelve None.
    """
    stats = getattr(gam, "statistics_", None)
    if stats is not None:
        return stats

    try:
        stats = gam.statistics()
        return stats
    except Exception:
        return None


def compute_concurvity(
    gam: GAM,
    X: np.ndarray,
    feature_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Estima la concurvity de cada término suave del GAM.

    Parámetros
    ----------
    gam : pygam.GAM o LinearGAM
        Modelo ya ajustado.
    X : np.ndarray, shape (n_muestras, n_predictores)
        Matriz de diseño usada para ajustar el modelo (mismas columnas y orden).
    feature_names : list[str], opcional
        Nombres de los predictores en el mismo orden que las columnas de X.
        Si es None, se generan nombres genéricos x0, x1, ...

    Devuelve
    --------
    df_conc : pd.DataFrame
        DataFrame con columnas:
        - 'feature': nombre del predictor
        - 'concurvity': [0, 1], aproximación a qué tanto ese término se puede
                        explicar a partir de los otros términos suaves.
                        0 ≈ término casi independiente; 1 ≈ altamente redundante.
        - 'source': 'pygam_statistics' si se usó la salida nativa de pygam,
                    'approx_r2' si se usó el método aproximado.
    """
    X = np.asarray(X)
    n_features = X.shape[1]

    if feature_names is None:
        feature_names = [f"x{i}" for i in range(n_features)]

    # --- 1) Intentar usar concurvity nativa de pygam, si existe ---
    stats = _safe_get_statistics(gam)
    if isinstance(stats, dict) and "concurv" in stats:
        conc = np.asarray(stats["concurv"]).ravel()

        # Ajustar longitud por seguridad
        if len(conc) != n_features:
            # Si la dimensionalidad no calza, mejor no usarla y pasar a approx
            pass
        else:
            df_native = pd.DataFrame(
                {
                    "feature": feature_names,
                    "concurvity": conc,
                    "source": "pygam_statistics",
                }
            )
            return df_native

    # --- 2) Método aproximado basado en R² entre términos suaves ---
    scores = []

    # Para cada predictor j:
    for j in range(n_features):
        # f_j(x) = contribución suave del término j
        f_j = gam.partial_dependence(X, feature=j)  # shape (n,)

        # contribuciones de los otros términos
        others = [
            gam.partial_dependence(X, feature=k)
            for k in range(n_features)
            if k != j
        ]

        if len(others) == 0:
            # Solo hay un término en el modelo; no tiene sentido hablar de concurvity
            scores.append(np.nan)
            continue

        F = np.column_stack(others)  # (n, n_features-1)

        # Añadimos intercepto para la regresión
        F_aug = np.column_stack([np.ones(F.shape[0]), F])

        # Ajuste OLS f_j ~ F_aug
        try:
            beta, *_ = np.linalg.lstsq(F_aug, f_j, rcond=None)
            f_j_hat = F_aug @ beta

            ss_res = np.sum((f_j - f_j_hat) ** 2)
            ss_tot = np.sum((f_j - f_j.mean()) ** 2)

            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        except Exception:
            r2 = np.nan

        scores.append(r2)

    df_approx = pd.DataFrame(
        {
            "feature": feature_names,
            "concurvity": scores,
            "source": "approx_r2",
        }
    )

    return df_approx
