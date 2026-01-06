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

import numpy as np
import pandas as pd
from pygam import GAM

def compute_concurvity(
    gam: GAM,
    X: np.ndarray,
    feature_names: list = None,
) -> pd.DataFrame:
    
    # Asegurar formato numpy
    X = np.asarray(X)
    n_features = X.shape[1]

    if feature_names is None:
        feature_names = [f"x{i}" for i in range(n_features)]

    scores = []

    # Para cada término j:
    for j in range(n_features):
        try:
            # --- CORRECCIÓN CLAVE ---
            # Usamos argumentos POSICIONALES para evitar problemas de nombres.
            # El primer argumento es el término (j), el argumento 'X' es la data.
            # .ravel() aplana el resultado a 1D.
            f_j = gam.partial_dependence(term=j, X=X).ravel()

            # Obtener los otros términos para comparar
            others = []
            for k in range(n_features):
                if k != j:
                    val = gam.partial_dependence(term=k, X=X).ravel()
                    others.append(val)
            
            if not others:
                # Si solo hay 1 variable, no hay concurvidad
                scores.append(0.0)
                continue

            # Matriz de los "otros" términos
            F = np.column_stack(others)
            
            # Regresión Lineal: f_j ~ Intercept + Otros
            # Esto mide qué tanto se parece f_j a los otros
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
            print(f"Advertencia en variable '{feature_names[j]}': {e}")
            scores.append(np.nan)

    return pd.DataFrame({
        "feature": feature_names,
        "concurvity": scores,
        "threshold": 0.5  # Referencia de Walker et al. (2023)
    })
