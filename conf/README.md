# tesis-s5p – Pipeline multigás S5P + ERA5 para detección de hotspots

Repositorio del pipeline desarrollado para la tesis de Maestría en Ciencia de Datos y Analítica (EAFIT).  
El objetivo es construir, a partir de **Sentinel-5P / TROPOMI (S5P)** y **ERA5**, un **ranking de celdas “hotspot”** donde es más eficiente priorizar inspecciones ambientales.

El enfoque es **multigás** (NO₂, CH₄, ozono troposférico, CO) y está estructurado siguiendo la lógica de **CRISP-DM**:

1. Construir un **AOI** (Area of Interest) a partir de GAUL.
2. Generar una **grilla regular** (fishnet) sobre el AOI.
3. Agregar mensualmente S5P y ERA5 sobre cada celda.
4. Ajustar modelos **GAM** meteorología → gas (por gas).
5. Calcular **anomalías robustas** (z-score + Mahalanobis) y un **score multigás** por celda.
6. Aplicar filtros espaciales (MMU) y **persistencia temporal**.
7. Evaluar el ranking frente a un “pseudo-inventario” de fuentes (GPPD).
8. Analizar la **importancia de covariables meteorológicas** con el método de **ghost variables** de Delicado & Peña (2023).

---

## 1. Estructura del repositorio

```text
tesis-s5p/
├─ conf/
│  └─ README.md          # (opcional) notas de configuración
├─ data/
│  ├─ raw/               # datos crudos / exportes desde Colab (gitkeep)
│  └─ processed/         # resultados procesados (gitkeep)
├─ notebooks/
│  └─ 01_run_aoi_grid.ipynb   # notebook principal del pipeline
├─ src/
│  ├─ __init__.py
│  ├─ config.py          # get_default_params(...): AOI, años, meses, gases, QA, grilla
│  ├─ geo_helpers.py     # construcción AOI (GAUL) + grilla (fishnet)
│  ├─ s5p_helpers.py     # inventario S5P y extracción mensual por gas
│  ├─ era5_helpers.py    # extracción y agregación mensual de ERA5
│  ├─ metrics.py         # HR–Área, PAI, AUC-PR, FDR@HR
│  ├─ df_inspecciones.py # construcción de labels (celdas con ≥1 planta GPPD)
│  ├─ ghost_variable.py  # análisis de variables fantasma (ghost variables)
│  └─ README.md          # (este archivo)
├─ .gitignore            # ignora datos pesados, checkpoints, etc.
└─ README.md

<img width="2029" height="1030" alt="Image" src="https://github.com/user-attachments/assets/130c6ceb-7d4a-469e-9945-c375251ccaff" />


