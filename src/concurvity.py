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

def _safe_get_statistics(gam: GAM):
    """Obtiene estadísticas si están disponibles."""
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

    # Para cada término j:
    for j in range(n_features):
        try:
            # CORRECCIÓN AQUÍ: Usamos 'term=j' en lugar de 'feature=j'
            # .ravel() asegura que sea un array 1D plano
            f_j = gam.partial_dependence(term=j, X=X).ravel()

            # Obtener los otros términos
            others = []
            for k in range(n_features):
                if k != j:
                    val = gam.partial_dependence(term=k, X=X).ravel()
                    others.append(val)
            
            if not others:
                scores.append(np.nan)
                continue

            # Matriz de los otros términos
            F = np.column_stack(others)
            
            # Regresión Lineal: f_j ~ Intercept + Otros
            F_aug = np.column_stack([np.ones(F.shape[0]), F])
            
            # Mínimos cuadrados
            beta, *_ = np.linalg.lstsq(F_aug, f_j, rcond=None)
            f_j_hat = F_aug @ beta

            ss_res = np.sum((f_j - f_j_hat) ** 2)
            ss_tot = np.sum((f_j - f_j.mean()) ** 2)

            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
            scores.append(r2)

        except Exception as e:
            print(f"Error calculando concurvity para {feature_names[j]}: {e}")
            scores.append(np.nan)

    return pd.DataFrame({
        "feature": feature_names,
        "concurvity": scores,
        "source": "approx_r2"
    })
