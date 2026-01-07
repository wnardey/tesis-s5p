# src/modelado.py
# -*- coding: utf-8 -*-
"""
Módulo central de modelado estadístico para la Tesis de Maestría.
Incluye:
1. Ajuste de Modelos GAM (LinearGAM).
2. Auditoría de independencia estructural (Concurvity).
3. Validación cruzada dual (Interpolación vs Generalización).
4. Generación de gráficos de diagnóstico.

Autor: Winston Salcedo
"""

# ==============================================================================
# 1. PARCHES DE COMPATIBILIDAD (CRÍTICO PARA PYTHON 3.12+)
# ==============================================================================
import sys
import types
import importlib

# Parche para librería 'imp' obsoleta requerida por pygam
if 'imp' not in sys.modules:
    fimp = types.ModuleType('imp')
    fimp.reload = importlib.reload
    sys.modules['imp'] = fimp

# ==============================================================================
# 2. IMPORTACIONES
# ==============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import r2_score, mean_squared_error
from pygam import LinearGAM, s

# ==============================================================================
# 3. CONFIGURACIÓN GLOBAL
# ==============================================================================

# Selección Minimalista Validada (Tesis Walker et al., 2023)
# Se excluyen u10, v10 y precip_mm por alta colinealidad.
PRED_COLS_MINIMAL = ['T2m_C', 'BLH', 'wind_speed']

META_INFO = {
    "NO2":    {"latex": r"$NO_2$",        "unit": r"mol/m$^2$", "color": "steelblue"},
    "O3_TCL": {"latex": r"$O_3$ (Trop)",  "unit": r"mol/m$^2$", "color": "forestgreen"},
    "CO":     {"latex": r"$CO$",          "unit": r"mol/m$^2$", "color": "darkorange"},
    "CH4":    {"latex": r"$CH_4$",        "unit": r"ppb",       "color": "firebrick"}
}

# ==============================================================================
# 4. FUNCIONES DE AJUSTE Y AUDITORÍA
# ==============================================================================

def compute_concurvity(gam, X, feature_names=None):
    """
    Calcula la concurvidad (dependencia no lineal) de los predictores.
    Retorna un DataFrame con índices entre 0 (indep) y 1 (redundante).
    """
    X = np.asarray(X)
    n_features = X.shape[1]
    if feature_names is None: feature_names = [f"x{i}" for i in range(n_features)]
    
    scores = []
    for j in range(n_features):
        try:
            # .ravel() asegura 1D array, usando 'term' en lugar de 'feature'
            f_j = gam.partial_dependence(term=j, X=X).ravel()
            
            others = []
            for k in range(n_features):
                if k != j:
                    others.append(gam.partial_dependence(term=k, X=X).ravel())
            
            if not others:
                scores.append(0.0)
                continue
                
            F = np.column_stack(others)
            F_aug = np.column_stack([np.ones(F.shape[0]), F]) # Intercepto
            
            # Regresión Lineal Auxiliar para estimar redundancia
            beta, _, _, _ = np.linalg.lstsq(F_aug, f_j, rcond=None)
            f_j_hat = F_aug @ beta
            
            ss_res = np.sum((f_j - f_j_hat) ** 2)
            ss_tot = np.sum((f_j - f_j.mean()) ** 2)
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
            scores.append(r2)
        except Exception:
            scores.append(0.0) # Fallback seguro
            
    return pd.DataFrame({"feature": feature_names, "concurvity": scores})

def ajustar_gam(df_ready, gas, pred_cols=PRED_COLS_MINIMAL):
    """
    Ajusta el modelo GAM, calcula residuales y Z-scores robustos (MAD).
    """
    df_gas = df_ready[df_ready["gas"] == gas].copy().reset_index(drop=True)
    if df_gas.empty: return None, None

    X = df_gas[pred_cols].values
    y = df_gas["value"].values
    y_log = np.log(np.clip(y, 1e-8, None))

    # Definición de splines
    terms = s(0, n_splines=6)
    for i in range(1, len(pred_cols)):
        terms += s(i, n_splines=6)

    # GridSearch para penalización optima
    gam = LinearGAM(terms).gridsearch(X, y_log, lam=[0.1, 1, 10, 100])

    # Predicciones
    y_hat = np.exp(gam.predict(X))
    resid = y - y_hat
    
    # Estandarización Robusta
    med_R = np.median(resid)
    mad_R = np.median(np.abs(resid - med_R))
    z_robust = (resid - med_R) / (1.4826 * max(mad_R, 1e-12))

    df_gas["y_obs"] = y
    df_gas["y_hat"] = y_hat
    df_gas["resid"] = resid
    df_gas["z_robust"] = z_robust

    return df_gas, gam

def validar_cv(df_ready, gas, pred_cols=PRED_COLS_MINIMAL, n_splits=5):
    """
    Ejecuta validación dual: KFold (Interpolación) vs GroupKFold (Extrapolación).
    """
    df_gas = df_ready[df_ready["gas"] == gas].copy().reset_index(drop=True)
    n_obs = len(df_gas)
    if n_obs < 15: return {}

    X = df_gas[pred_cols].values
    y_log = np.log(np.clip(df_gas["value"].values, 1e-8, None))
    
    terms = s(0, n_splines=6)
    for i in range(1, len(pred_cols)): terms += s(i, n_splines=6)
    
    # 1. KFold Aleatorio
    kf = KFold(n_splits=min(n_splits, n_obs), shuffle=True, random_state=42)
    r2_k = []
    for train, test in kf.split(X):
        gam = LinearGAM(terms).gridsearch(X[train], y_log[train], lam=[0.1, 1, 10])
        r2_k.append(r2_score(np.exp(y_log[test]), np.exp(gam.predict(X[test]))))

    # 2. GroupKFold (Temporal)
    groups = df_gas["year"].astype(str) + "-" + df_gas["month"].astype(str)
    r2_g = []
    if len(groups.unique()) >= 2:
        gkf = GroupKFold(n_splits=min(n_splits, len(groups.unique())))
        for train, test in gkf.split(X, y_log, groups=groups):
            gam = LinearGAM(terms).gridsearch(X[train], y_log[train], lam=[0.1, 1, 10])
            r2_g.append(r2_score(np.exp(y_log[test]), np.exp(gam.predict(X[test]))))
            
    return {
        'r2_k_m': np.mean(r2_k), 'r2_k_s': np.std(r2_k),
        'r2_g_m': np.mean(r2_g) if r2_g else np.nan, 
        'r2_g_s': np.std(r2_g) if r2_g else np.nan
    }

def plot_diagnosticos(resultados_dict):
    """Genera el panel 4x2 de Scatter plots e Histogramas."""
    gases = list(resultados_dict.keys())
    fig, axes = plt.subplots(nrows=len(gases), ncols=2, figsize=(12, 4*len(gases)))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    
    if len(gases) == 1: axes = [axes] # Manejo si solo hay 1 gas

    for i, gas in enumerate(gases):
        df_g = resultados_dict[gas]["df"]
        info = META_INFO.get(gas, {"latex": gas, "unit": "u.a.", "color": "gray"})
        
        # Métricas
        r2 = r2_score(df_g["y_obs"], df_g["y_hat"])
        rmse = np.sqrt(mean_squared_error(df_g["y_obs"], df_g["y_hat"]))
        
        # Columna 1: Scatter
        ax1 = axes[i][0]
        ax1.scatter(df_g["y_hat"], df_g["y_obs"], alpha=0.6, color=info['color'], edgecolor='k')
        m_val = max(df_g["y_hat"].max(), df_g["y_obs"].max())
        ax1.plot([0, m_val], [0, m_val], "k--", alpha=0.5)
        
        # Caja de texto
        stats = f"$R^2={r2:.2f}$\n$RMSE={rmse:.2e}$"
        ax1.text(0.05, 0.95, stats, transform=ax1.transAxes, va='top', 
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax1.set_title(f"{info['latex']} - Bondad de Ajuste")
        ax1.set_xlabel(f"Predicción [{info['unit']}]")
        ax1.set_ylabel(f"Observado [{info['unit']}]")
        ax1.grid(True, ls=':', alpha=0.5)
        
        # Columna 2: Histograma
        ax2 = axes[i][1]
        z_vals = df_g["z_robust"]
        ax2.hist(z_vals, bins=15, density=True, color=info['color'], alpha=0.6, edgecolor='k')
        x = np.linspace(z_vals.min(), z_vals.max(), 100)
        ax2.plot(x, norm.pdf(x), 'k-', lw=1.5, label=r'$\mathcal{N}(0,1)$')
        
        ax2.set_title(f"{info['latex']} - Residuales Normalizados")
        ax2.set_xlabel(r"Z-score ($z$)")
        ax2.grid(True, ls=':', alpha=0.5)

    return fig
