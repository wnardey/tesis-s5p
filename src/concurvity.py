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

"""
Módulo para auditoría de independencia estructural (concurvity) en modelos GAM.
Implementa la validación metodológica según Walker et al. (2023).
"""

import numpy as np
import pandas as pd
from pygam import GAM

def _safe_get_statistics(gam: GAM):
    """
    Intenta obtener estadísticas nativas del GAM si están disponibles.
    Maneja diferencias de versión en pygam.
    """
    stats = getattr(gam, "statistics_", None)
    if stats is not None:
        return stats
    try:
        return gam.statistics()
    except Exception:
        return None

def compute_concurvity(
    gam: GAM,
    X: np.ndarray,
    feature_names: list = None,
) -> pd.DataFrame:
    """
    Calcula la concurvidad (dependencia no lineal) de los predictores.
    
    Si pygam no expone la estadística nativa, estima una aproximación 
    post-hoc calculando el R^2 de cada término suave frente al resto.
    
    Returns:
        pd.DataFrame: Tabla con índices de concurvidad (0=Indep, 1=Redundante).
    """
    # Asegurar formato numpy y dimensiones
    X = np.asarray(X)
    n_features = X.shape[1]

    if feature_names is None:
        feature_names = [f"x{i}" for i in range(n_features)]

    # --- 1) Intentar usar concurvity nativa de pygam ---
    stats = _safe_get_statistics(gam)
    if isinstance(stats, dict) and "concurv" in stats:
        conc = np.asarray(stats["concurv"]).ravel()
        if len(conc) == n_features:
            return pd.DataFrame({
                "feature": feature_names,
                "concurvity": conc,
                "source": "pygam_statistics"
            })

    # --- 2) Método aproximado (R^2 entre términos) ---
    scores = []

    for j in range(n_features):
        try:
            # Obtener la contribución del término j (f_j)
            # .ravel() es crucial para asegurar 1D array
            f_j = gam.partial_dependence(term=j, X=X).ravel()

            # Obtener contribuciones de los otros términos (f_-j)
            others = []
            for k in range(n_features):
                if k != j:
                    val = gam.partial_dependence(term=k, X=X).ravel()
                    others.append(val)
            
            if not others:
                scores.append(0.0)
                continue

            # Matriz de diseño de los "otros" términos
            F = np.column_stack(others)
            
            # Regresión Lineal Auxiliar: f_j ~ Intercept + f_others
            F_aug = np.column_stack([np.ones(F.shape[0]), F])
            
            # Resolver mínimos cuadrados (OLS)
            beta, _, _, _ = np.linalg.lstsq(F_aug, f_j, rcond=None)
            f_j_hat = F_aug @ beta

            # Calcular R^2 de esta relación
            ss_res = np.sum((f_j - f_j_hat) ** 2)
            ss_tot = np.sum((f_j - f_j.mean()) ** 2)

            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
            scores.append(r2)

        except Exception as e:
            # En caso de error (ej. término no suave), reportar NaN
            scores.append(np.nan)

    return pd.DataFrame({
        "feature": feature_names,
        "concurvity": scores,
        "threshold": 0.5  # Criterio Walker et al. (2023)
    })
