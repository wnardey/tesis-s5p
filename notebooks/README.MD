# tesis-s5p

Pipeline de detección de hotspots de contaminantes atmosféricos a partir de datos Sentinel-5P (S5P/TROPOMI) y ERA5, ejecutado íntegramente en **Google Colab** sobre **Google Earth Engine (GEE)**.

---

## Tabla de contenidos

1. [Descripción general](#descripción-general)
2. [Arquitectura del pipeline](#arquitectura-del-pipeline)
3. [Estructura del repositorio](#estructura-del-repositorio)
4. [Requisitos previos](#requisitos-previos)
5. [Paso 1 — Crear un Cloud Project en GEE](#paso-1--crear-un-cloud-project-en-gee)
6. [Paso 2 — Generar un token de GitHub](#paso-2--generar-un-token-de-github)
7. [Paso 3 — Variables a modificar en el notebook](#paso-3--variables-a-modificar-en-el-notebook)
8. [Paso 4 — Ejecutar el notebook en Colab](#paso-4--ejecutar-el-notebook-en-colab)
9. [Salidas generadas](#salidas-generadas)
10. [Dependencias](#dependencias)

---

## Descripción general

El notebook `01_run_aoi_grid.ipynb` implementa un flujo de análisis espacial multigás para identificar celdas de cuadrícula con anomalías persistentes de contaminantes (NO₂, CO, O₃ troposférico y CH₄), usando:

- **Sentinel-5P / TROPOMI** — columnas troposféricas de gases traza.
- **ERA5** — variables meteorológicas (temperatura, viento, BLH, precipitación).
- **Modelos GAM** — ajuste suavizado por gas con validación cruzada espacial y temporal.
- **Score multigás (distancia de Mahalanobis robusta, MCD)** — combina las anomalías Z por gas en un único índice.
- **Evaluación vs. GPPD** — métricas HR–Área, PAI y AUC–PR frente al inventario global de plantas de energía.

El análisis corre sobre **Google Colab** y accede a GEE a través de la API oficial de Python.

---

## Arquitectura del pipeline

```
Colab Runtime
│
├── 1. Configuración del entorno
│   ├── pip install earthengine-api geemap pygam
│   ├── Clonar repositorio desde GitHub (token requerido)
│   └── Inicializar Earth Engine (Cloud Project requerido)
│
├── 2. Entendimiento de los datos
│   ├── ED_01 – AOI + cuadrícula + mapa diagnóstico
│   ├── ED_02 – Inventario S5P multigás
│   └── ED_03 – Inventario ERA5 mensual
│
├── 3. Procesamiento
│   ├── PD_01 – df_base: S5P + ERA5 por celda/mes
│   └── PD_02 – Limpieza, filtro QA y enriquecimiento → df_ready
│
├── 4. Modelado
│   ├── GAM por gas (NO2, O3_TCL, CO, CH4)
│   ├── Auditoría de concurvidad (independencia estructural)
│   ├── Validación cruzada k-fold y temporal (GroupKFold)
│   ├── Matriz multigás Z → score S (Mahalanobis robusto)
│   └── Umbral P90 + filtro MMU (mínima unidad mapeada)
│
└── 5. Evaluación
    ├── Mapa de hotspots persistentes (espacio-temporal)
    ├── Curva HR–Área y PAI vs. GPPD
    ├── AUC–PR
    └── FDR@HR=0.8 / FDR@HR=0.9
```

---

## Estructura del repositorio

```
tesis-s5p/
├── 01_run_aoi_grid.ipynb   # Notebook principal (ejecutar en Colab)
├── src/
│   ├── config.py           # get_default_params (AOI, años, meses, escala)
│   ├── geo_helpers.py      # build_aoi, build_square_grid
│   ├── s5p_helpers.py      # inventario_s5p, GAS_SPECS, choose_ic
│   ├── era5_helpers.py     # era5_monthly, band_mean
│   ├── metrics.py          # build_eval_df, compute_hr_area_pai, compute_auc_pr, fdr_at_hr
│   ├── modelado.py         # ajustar_gam, validar_cv, compute_concurvity, plot_diagnosticos
│   ├── df_inspecciones.py  # build_df_inspecciones_gppd
│   └── ghost_variable.py   # run_full_analysis
└── data/
    └── processed/          # CSVs generados por el notebook (creado en runtime)
```

---

## Requisitos previos

| Requisito | Descripción |
|-----------|-------------|
| Cuenta de Google | Necesaria para acceder a Colab y GEE |
| Cloud Project de GEE | Proyecto en Google Cloud con la API de Earth Engine habilitada |
| Token de GitHub | Personal Access Token con scope `repo` para clonar el repositorio privado |

---

## Paso 1 — Crear un Cloud Project en GEE

Google Earth Engine exige que cada llamada a la API esté asociada a un **Google Cloud Project** registrado.

### 1.1 Crear el proyecto en Google Cloud

1. Ve a [console.cloud.google.com](https://console.cloud.google.com) e inicia sesión con tu cuenta de Google.
2. En la barra superior, haz clic en el selector de proyectos → **"Nuevo proyecto"**.
3. Asigna un nombre descriptivo (p. ej. `tesis-s5p-gee`) y anota el **ID del proyecto** (se genera automáticamente, p. ej. `tesis-s5p-gee-123456`).
4. Haz clic en **"Crear"**.

### 1.2 Habilitar la API de Earth Engine

1. En el menú lateral izquierdo ve a **"APIs y servicios" → "Biblioteca"**.
2. Busca **"Earth Engine API"** y haz clic en **"Habilitar"**.

### 1.3 Registrar el proyecto en GEE

1. Ve a [code.earthengine.google.com/register](https://code.earthengine.google.com/register) o a [earthengine.google.com/signup](https://earthengine.google.com/signup).
2. Selecciona **"Use with a Cloud Project"** e ingresa el ID de tu proyecto.
3. Acepta los términos de servicio.
4. La solicitud de acceso puede tardar unas horas en aprobarse (normalmente es instantánea para cuentas educativas o de investigación).

> **Nota:** El ID de proyecto tiene la forma `nombre-palabras-123456`. Necesitarás este valor para configurar el notebook.

---

## Paso 2 — Generar un token de GitHub

El notebook clona el repositorio directamente desde GitHub dentro de Colab usando un **Personal Access Token (PAT)**.

### 2.1 Crear el token

1. Inicia sesión en [github.com](https://github.com).
2. Ve a **Settings** (esquina superior derecha, tu avatar) → **Developer settings** → **Personal access tokens** → **Tokens (classic)**.
3. Haz clic en **"Generate new token (classic)"**.
4. Configura:
   - **Note:** `colab-tesis-s5p` (o cualquier nombre descriptivo)
   - **Expiration:** elige el período que necesites (p. ej. 90 días)
   - **Scopes:** activa **`repo`** (incluye lectura y escritura de repositorios privados)
5. Haz clic en **"Generate token"**.
6. **Copia el token inmediatamente** — GitHub solo lo muestra una vez. Guárdalo en un lugar seguro (gestor de contraseñas).

### 2.2 Uso del token en Colab

Cuando el notebook se ejecute pedirá primero el usuario y luego el token:

```
GitHub username: ··········
GitHub Token (repo scope): ··········
```

Pega el token y pulsa Enter. Colab no lo almacena; deberás ingresarlo cada vez que inicies un nuevo runtime. Además, el notebook reescribe el remote URL después del clone para que el token nunca quede visible en outputs posteriores.

> **Seguridad:** Nunca incluyas el token directamente en el código del notebook ni lo subas al repositorio. El notebook usa `getpass.getpass()` precisamente para evitarlo. Antes de cada commit ejecuta **Kernel → Restart and clear output**.

---

## Paso 3 — Variables a modificar en el notebook

Abre `01_run_aoi_grid.ipynb` en Colab. Solo hay **una variable fija** que debes editar; el resto se ingresa de forma segura en tiempo de ejecución.

### Celda 5 — Clonar el repositorio

El notebook solicita credenciales de forma interactiva mediante `getpass`:

```
GitHub username: ··········
GitHub Token (repo scope): ··········
```

Solo debes cambiar el nombre del repositorio si lo renombraste:

```python
REPO = "tesis-s5p"   # <-- cambia si renombraste el repositorio
```

Después del clone, el remote URL se reescribe automáticamente para eliminar las credenciales, por lo que `git remote -v` nunca imprimirá el token:

```python
!git remote set-url origin https://github.com/{USER}/{REPO}.git
```

### Celda 9 — Inicializar Earth Engine

El notebook solicita el ID del Cloud Project de forma interactiva:

```
GEE Cloud Project ID: ··········
```

Ningún secreto queda escrito en el código.

### Celda 12 — Parámetros temporales (opcional)

```python
YEARS  = [2023, 2024]   # años de análisis
MONTHS = [5, 6, 7]      # meses de análisis (mayo–julio)
```

Modifica `YEARS` y `MONTHS` si deseas analizar un período diferente. El resto de parámetros espaciales (AOI, resolución de cuadrícula, etc.) se controlan desde `src/config.py`.

---

## Paso 4 — Ejecutar el notebook en Colab

1. Sube el notebook a [colab.research.google.com](https://colab.research.google.com) o ábrelo directamente desde GitHub con el botón **"Open in Colab"**.
2. Ejecuta las celdas en orden de arriba hacia abajo.
3. En la **Celda 5** ingresa tu usuario y token de GitHub cuando se soliciten.
4. En la **Celda 9** ingresa el ID de tu Cloud Project de GEE. Si es la primera vez que usas GEE en este runtime, aparecerá un enlace de autenticación:
   - Haz clic en el enlace.
   - Selecciona tu cuenta de Google.
   - Autoriza el acceso a Earth Engine y Google Drive.
   - Copia el código de verificación y pégalo en Colab.

> **Importante:** Antes de hacer commit o push del notebook, ejecuta **Kernel → Restart and clear output** para asegurarte de que ningún output de celda contenga credenciales o URLs de autenticación.

El pipeline completo puede tardar entre **20 y 60 minutos** según el período analizado, la velocidad de la API de GEE y los recursos del runtime de Colab.

---

## Salidas generadas

El notebook guarda tres archivos CSV en `data/processed/` dentro del repositorio clonado:

| Archivo | Descripción |
|---------|-------------|
| `df_ready_AOI_5km_2023_2024.csv` | Dataset limpio: S5P + ERA5 por celda y mes |
| `df_z_wide_scores.csv` | Matriz multigás Z + score de Mahalanobis (S_multigas) |
| `df_persistencia_2023_2024.csv` | Resumen de persistencia por celda (hotspots espacio-temporales) |

Estos archivos se pueden subir al repositorio ejecutando las celdas de commit/push del notebook (requiere que el token de GitHub tenga permisos de escritura).

---

## Dependencias

El notebook instala automáticamente las bibliotecas necesarias en el runtime de Colab:

```bash
pip install earthengine-api geemap --upgrade
pip install pygam
```

Bibliotecas del entorno estándar de Colab también utilizadas:

| Biblioteca | Uso |
|-----------|-----|
| `numpy`, `pandas` | Manipulación de datos |
| `matplotlib`, `seaborn` | Visualización |
| `scikit-learn` | KFold, GroupKFold, métricas, MinCovDet |
| `scipy` | Distribución normal |
| `ee` (earthengine-api) | Acceso a GEE |
| `geemap` | Mapas interactivos sobre GEE |
| `pygam` | Modelos aditivos generalizados (GAM) |

---

## Advertencias de seguridad

- **Nunca** incluyas tu token de GitHub ni el ID de proyecto de GEE en texto plano dentro del notebook si este es un repositorio público.
- El notebook usa `getpass.getpass()` para solicitar el token de forma segura en cada sesión.
- Si crees que tu token fue expuesto, revócalo inmediatamente en **GitHub → Settings → Developer settings → Personal access tokens**.
