import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pygam import LinearGAM
from typing import List, Optional, Tuple, Dict

class GhostVariableAnalyzer:
    """
    Implementación del método de Variables Fantasma (Ghost Variables) 
    para interpretabilidad de modelos predictivos.
    
    Referencia: Delicado, P., & Peña, D. (2023). Understanding complex 
    predictive models with ghost variables.
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def _fit_gam(self, X: np.ndarray, y: np.ndarray) -> Tuple[LinearGAM, float, float]:
        """Ajusta un modelo GAM y retorna el objeto modelo, el R2 y la suma total de cuadrados."""
        try:
            gam = LinearGAM(verbose=self.verbose).fit(X, y)
            y_pred = gam.predict(X)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            
            # Evitar división por cero
            if ss_tot == 0:
                return gam, np.nan, 0.0
                
            r2 = 1.0 - (ss_res / ss_tot)
            return gam, r2, ss_tot
        except Exception:
            return None, np.nan, 0.0

    def _get_ghost_feature(self, X_rest: np.ndarray, z_target: np.ndarray) -> np.ndarray:
        """Genera la variable fantasma prediciendo Z en función del resto de variables (X_rest)."""
        try:
            # Ajustamos modelo auxiliar Z ~ X_rest
            gam_aux, r2, _ = self._fit_gam(X_rest, z_target)
            if gam_aux is None:
                return np.full_like(z_target, z_target.mean())
            return gam_aux.predict(X_rest)
        except:
            # Fallback a la media si falla el ajuste auxiliar
            return np.full_like(z_target, z_target.mean())

    def compute_importance(
        self,
        df: pd.DataFrame,
        features: List[str],
        target: str,
        log_transform: bool = True
    ) -> pd.DataFrame:
        """
        Calcula la importancia de variables para un dataset específico.
        """
        # 1. Preprocesamiento seguro
        cols = features + [target]
        sub = df[cols].replace([np.inf, -np.inf], np.nan).dropna()

        if len(sub) < 30: # Umbral mínimo de seguridad estadística
            return pd.DataFrame()

        X = sub[features].to_numpy(dtype=float)
        y = sub[target].to_numpy(dtype=float)

        if log_transform:
            y = np.log(np.clip(y, 1e-12, None))

        # 2. Modelo Base (Y ~ X)
        main_model, base_r2, ss_tot = self._fit_gam(X, y)
        
        if np.isnan(base_r2):
            return pd.DataFrame()

        results = []

        # 3. Iteración Ghost
        for j, var_name in enumerate(features):
            # Aislar variable objetivo Z y predictores restantes
            z_target = X[:, j]
            X_rest = np.delete(X, j, axis=1)

            if X_rest.shape[1] == 0:
                continue

            # Generar Ghost Variable (E[Z|X])
            z_ghost = self._get_ghost_feature(X_rest, z_target)

            # Crear matriz fantasma (reemplazando la original por la predicha)
            X_ghost = X.copy()
            X_ghost[:, j] = z_ghost

            # Predecir con el modelo base original (sin reentrenar)
            y_pred_ghost = main_model.predict(X_ghost)

            # Calcular métrica de impacto
            ss_res_ghost = np.sum((y - y_pred_ghost) ** 2)
            ghost_r2 = 1.0 - (ss_res_ghost / ss_tot)
            
            # Delta R2: Cuánto rendimiento pierde el modelo al usar la versión "fantasma"
            delta_r2 = base_r2 - ghost_r2

            results.append({
                "var": var_name,
                "delta_r2": max(delta_r2, 0.0) # Clipping a 0 para visualización limpia
            })

        return pd.DataFrame(results).sort_values("delta_r2", ascending=False)

def plot_heatmap(results_dict: Dict[str, pd.DataFrame], features: List[str]):
    """Genera el mapa de calor normalizado a partir de los resultados."""
    if not results_dict:
        print("No hay resultados para graficar.")
        return

    gases = list(results_dict.keys())
    matrix = np.zeros((len(features), len(gases)))

    # Llenar matriz
    for j, gas in enumerate(gases):
        df = results_dict[gas]
        if df.empty: continue
        
        # Normalización Min-Max por columna (por gas) para comparabilidad
        max_val = df["delta_r2"].max()
        if max_val > 0:
            for i, feat in enumerate(features):
                row = df[df["var"] == feat]
                if not row.empty:
                    val = row["delta_r2"].iloc[0]
                    matrix[i, j] = val / max_val

    # Plotting
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis", origin="upper") # Viridis o Magma

    # Decoración
    ax.set_xticks(np.arange(len(gases)))
    ax.set_yticks(np.arange(len(features)))
    ax.set_xticklabels([g.replace("_TCL", "") for g in gases], weight='bold')
    ax.set_yticklabels(features)
    ax.set_title("Importancia Relativa de Variables (Método Ghost)\nDelta R² Normalizado", pad=20)

    # Anotaciones
    for i in range(len(features)):
        for j in range(len(gases)):
            val = matrix[i, j]
            color = "white" if val < 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=9)

    plt.colorbar(im, label="Importancia Relativa (0-1)")
    plt.tight_layout()
    plt.show()

def run_full_analysis(df: pd.DataFrame, features: List[str], gases: List[str]):
    """Orquestador principal que imprime los logs requeridos y genera el gráfico."""
    
    print("Iniciando análisis de Variables Fantasma...\n")
    analyzer = GhostVariableAnalyzer()
    results_store = {}

    for gas in gases:
        # Filtrar datos para el gas actual
        df_gas = df[df["gas"] == gas].copy()
        n_rows = len(df_gas)
        
        print(f"Analizando {gas} ({n_rows} filas)...")
        
        if n_rows < 30:
            print(" -> Saltando por insuficiencia de datos.\n")
            continue

        # Cálculo
        df_imp = analyzer.compute_importance(df_gas, features, target="value")
        
        if not df_imp.empty:
            top_var = df_imp.iloc[0]
            print(f" -> Completado. Variable más importante: {top_var['var']} (Delta R2: {top_var['delta_r2']:.4f})\n")
            results_store[gas] = df_imp
        else:
            print(" -> No se pudo converger o R2 negativo.\n")

    # Visualización final
    plot_heatmap(results_store, features)
