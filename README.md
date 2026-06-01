# Sistema Predictivo de Saturacion en Urgencias

> **Version 0.3** — 5 niveles de severidad operativa · 3 horizontes de prediccion independientes (+1h, +3h, +6h)
>
> Datos reales de Nuestra Cali (2024-2026) · Dashboard para coordinadores medicos

---

## Tabla de contenidos

1. [El problema](#1-el-problema)
2. [La solucion](#2-la-solucion)
3. [Captura del dashboard](#3-captura-del-dashboard)
4. [Arquitectura del sistema](#4-arquitectura-del-sistema)
5. [Estructura del repositorio](#5-estructura-del-repositorio)
6. [Requisitos](#6-requisitos)
7. [Instalacion (one-click)](#7-instalacion-one-click)
8. [Uso del dashboard](#8-uso-del-dashboard)
9. [Modelo predictivo](#9-modelo-predictivo)
10. [Resultados y metricas](#10-resultados-y-metricas)
11. [Limitaciones conocidas](#11-limitaciones-conocidas)
12. [Licencia](#12-licencia)
13. [Hoja de ruta](#13-hoja-de-rura)

---

## 1. El problema

Los servicios de urgencias en Colombia operan frecuentemente por encima del **120% de su capacidad instalada**. Esta saturacion no es un evento puntual, sino un estado cronico que genera:

- **Demoras en triage** que violan los tiempos establecidos por la Resolucion 256 de 2016
- **Abandono de pacientes** antes de ser atendidos
- **Fatiga y error humano** en el personal medico y de enfermeria
- **Riesgos legales** para la IPS por incumplimiento de estandares de calidad

Los coordinadores medicos y jefes de enfermeria enfrentan **"ceguera operativa"**: gestionan el censo de camas minuto a minuto de forma reactiva, sin poder anticipar los picos de demanda antes de que el servicio colapse.

---

## 2. La solucion

Implementamos **3 modelos de Machine Learning independientes** (Random Forest) que analizan el estado en tiempo real del servicio para predecir el nivel de severidad operativa en **3 horizontes**: +1 hora, +3 horas y +6 horas.

### 5 niveles de severidad operativa

| Nivel | Rango | % del tiempo | Color | Accion recomendada |
|---|---|---|---|---|
| Verde_Amarillo | < 1.0 | 4% | 🟢 | Operacion normal · Mantener monitoreo |
| Presion_Moderada | 1.0 - 1.8 | 26% | 🟡 | Preparar refuerzos · Revisar altas |
| Presion_Alta | 1.8 - 2.8 | 47% | 🟠 | Activar protocolo preventivo |
| Critico | 2.8 - 3.8 | 21% | 🔴 | Activar contingencia · Movilizar personal |
| Colapso | > 3.8 | 3% | 🟣 | Activar plan de crisis hospitalario |

Los umbrales estan basados en la **distribucion real del dataset** (percentiles ~4%, 30%, 77%, 97%).

### 3 horizontes de prediccion

| Horizonte | Balanced Accuracy | F1-macro | Utilidad operativa |
|---|---|---|---|
| **+1h** | **0.880** | **0.888** | Reaccion inmediata |
| **+3h** | **0.781** | **0.818** | Planificacion optima (recomendado) |
| **+6h** | **0.745** | **0.791** | Vision estrategica del turno |

---


## 4. Arquitectura del sistema

```
Power BI / Microsoft Fabric
       │
       ▼
  extraer_datos.ps1 (DAX queries)
       │
       ▼
  [triage_metricas_horarias.csv, triage_timestamps.csv, urgencias_timestamps.csv]
       │
       ▼
  preparar_dataset.py (ETL · prevencion de data leakage)
       │
       ▼
  dataset_train.csv  (20,667 filas · 24 columnas)
       │
       ▼
  retrain_model.py  /  modelo_severidad_urgencias_v3.ipynb
       │
       ├──► modelo_rf_1h.pkl   (Random Forest +1 hora)
       ├──► modelo_rf_3h.pkl   (Random Forest +3 horas)
       ├──► modelo_rf_6h.pkl   (Random Forest +6 horas)
       ├──► config_v3.pkl      (Umbrales, niveles, colores)
       └──► features_v3.pkl    (Lista de 24 features)
       │
       ▼
  src/app.py  (Streamlit Dashboard)
       │
       ▼
  Navegador web (http://localhost:8501)
```

### Pipeline de datos

1. **Extraccion**: Power BI → DAX queries → archivos CSV
2. **ETL**: `preparar_dataset.py` limpia, alinea temporalmente y genera lags (sin data leakage)
3. **Feature engineering**: encoding ciclico (seno/coseno) para hora y dia, presion combinada del sistema
4. **Entrenamiento**: 3 Random Forest independientes con validacion cruzada temporal (TimeSeriesSplit, 5 folds)
5. **Dashboard**: Streamlit carga los modelos y predictores, muestra estado actual + predicciones

---

## 5. Estructura del repositorio

```
saturacion-urgencias/
│
├── src/
│   ├── app.py                       # Dashboard Streamlit (758 lines)
│   ├── preparar_dataset.py          # ETL pipeline
│   ├── extraer_datos.ps1            # Extraccion Power BI / Fabric
│   └── queries_extraccion.md        # Documentacion de queries DAX
│
├── notebooks/
│   ├── modelo_severidad_urgencias_v3.ipynb  # Notebook v3 (33 celdas)
│   ├── modelo_saturacion_urgencias_v2.ipynb  # Notebook v2 (historico)
│   ├── modelo_saturacion_urgencias.ipynb     # Notebook v1 (historico)
│   ├── retrain_model.py                     # Script de reentrenamiento
│   └── build_v3.py                          # Generador del notebook
│
├── data/                             # ★ Datos (en .gitignore)
│   └── dataset_train.csv             #   20,667 filas
│
├── models/                           # ★ Modelos (en .gitignore)
│   ├── modelo_rf_1h.pkl              #   RF +1h (~50 MB)
│   ├── modelo_rf_3h.pkl              #   RF +3h (~50 MB)
│   ├── modelo_rf_6h.pkl              #   RF +6h (~50 MB)
│   ├── config_v3.pkl                 #   Umbrales y configuracion
│   └── features_v3.pkl               #   Lista de features
│
├── setup.sh                          # Instalacion one-click
├── pyproject.toml                    # Dependencias Python
├── uv.lock                           # Lockfile reproducible
├── .gitignore
├── LICENSE                           # CC BY-NC 4.0
└── README.md                         # Este archivo
```

> **★ Datos y modelos no estan incluidos en el repositorio** por peso. Consulta la seccion [Instalacion](#7-instalacion-one-click) para generarlos.

---

## 6. Requisitos

- **Python 3.10 o superior**
- **UV** (gestor de paquetes Python) — [Instalar UV](https://docs.astral.sh/uv/)
- **Git Bash** (Windows) o terminal bash (Linux/macOS)
- ~2 GB de espacio en disco (modelos incluidos)
- Navegador web moderno (Chrome, Firefox, Edge)

### Dependencias principales

| Paquete | Version minima | Uso |
|---|---|---|
| streamlit | >= 1.32.0 | Dashboard web |
| pandas | >= 2.0.0 | Manipulacion de datos |
| numpy | >= 1.24.0 | Computacion numerica |
| scikit-learn | >= 1.3.0 | Random Forest + metricas |
| plotly | >= 5.18.0 | Graficos interactivos |
| matplotlib | >= 3.7.0 | Graficos estaticos (notebook) |
| seaborn | >= 0.12.0 | Visualizacion (notebook) |

---

## 7. Instalacion (one-click)

### Paso 1: Clonar el repositorio

```bash
git clone https://github.com/tu-usuario/saturacion-urgencias.git
cd saturacion-urgencias
```

### Paso 2: Ejecutar el setup

```bash
# En Git Bash (Windows), Linux o macOS:
bash setup.sh
```

El script:
1. Verifica que UV este instalado (si no, lo instala)
2. Ejecuta `uv sync` para instalar todas las dependencias
3. Pregunta si deseas entrenar los modelos (recomendado: si)
   - El entrenamiento toma ~2-5 minutos
   - Genera 3 modelos en `models/`

### Paso 3: Iniciar el dashboard

```bash
uv run streamlit run src/app.py
```

O con el entorno virtual activado:

```bash
# Linux/macOS
source .venv/bin/activate
streamlit run src/app.py

# Windows (PowerShell)
.venv\Scripts\activate
streamlit run src/app.py
```

Esto abrira automaticamente el navegador en `http://localhost:8501`.

---

## 8. Uso del dashboard

### Panel principal

| Seccion | Que muestra | Cuando usarla |
|---|---|---|
| **Banner de severidad** | Estado actual, indice, tendencia, accion recomendada | Al abrir el dashboard (vista principal) |
| **Metricas** | Cola, camas, triage, abandonos con delta | Evaluacion rapida del turno |
| **Semafaro** | Nivel de severidad con indice numerico | Identificar el estado de un vistazo |
| **Grafica 12h + prediccion** | Historico reciente + proyeccion 3 horizontes | Antes de cambio de guardia |
| **Tarjetas de prediccion** | +1h, +3h, +6h con barra, badge y tendencia | Planificar refuerzos por hora |
| **KPIs de calidad** | 5 indicadores con estado OK/En riesgo/Bajo | Reporte a coordinacion |
| **Heatmap historico** | Saturacion promedio por hora y dia de semana | Planificacion de turnos y recursos |

### Sidebar

En el panel izquierdo puedes:
- Seleccionar una **fecha** y **hora** para simular un turno historico
- Ver el estado que el dashboard hubiera mostrado en ese momento

### Interpretacion de la tendencia

| Flecha | Significado | Accion sugerida |
|---|---|---|
| ⬆ Al alza | La saturacion esta aumentando | Preparar recursos adicionales |
| ➡ Estable | La saturacion se mantiene | Mantener monitoreo |
| ⬇ A la baja | La saturacion esta disminuyendo | Relajar protocolos si aplica |

---

## 9. Modelo predictivo

### Algoritmo

**Random Forest Regressor** — 3 modelos independientes (uno por horizonte).

Cada modelo usa 300 arboles, profundidad maxima 12 y min_samples_leaf=5.

### Features (24 total)

**Base (18):**
- Cantidad de triage, cola de espera, camas ocupadas, abandonos, capacidad
- Lags de triage (1h, 2h, 3h) y saturacion (1h, 3h, 6h)
- Lags de abandonos, cola, camas
- Tiempos de espera (turnero, consulta)
- Porcentajes de cumplimiento (oportunidad triage, consulta T2)
- Indicador de festivo

**Encoding ciclico (4):**
- `hora_sin`, `hora_cos` — representacion seno/coseno de la hora (0-23)
- `dia_sin`, `dia_cos` — representacion seno/coseno del dia de la semana (0-6)

Esto evita el salto artificial 23h → 0h y Dom → Lun.

**Presion de sistema (1):**
- `presion_sistema = cola + camas_ocupadas` — senal unificada de tension

### Validacion

**TimeSeriesSplit** con 5 folds (sin shuffle) para respetar el orden temporal.

### Metricas

| Metrica | Que mide | Por que es importante |
|---|---|---|
| **MASE** | Error escalado contra naive (persistencia) | < 1.0 significa que el modelo supera a "no hay cambio" |
| **Balanced Accuracy** | Promedio del recall por clase | No se infla por clase mayoritaria |
| **F1-macro** | Promedio de F1 por clase | Equilibrio precision-recall en todas las clases |

---

## 10. Resultados y metricas

### Validacion cruzada (5 folds)

| Horizonte | MAE | RMSE | R² | MASE |
|---|---|---|---|---|
| +1h | 0.161 | 0.195 | 0.880 | 1.47 |
| +3h | 0.218 | 0.270 | 0.824 | 2.07 |
| +6h | 0.269 | 0.335 | 0.754 | 2.63 |

### Clasificacion (5 niveles) — test holdout

| Horizonte | Balanced Accuracy | F1-macro |
|---|---|---|
| **+1h** | **0.880** | **0.888** |
| +3h | 0.781 | 0.818 |
| +6h | 0.745 | 0.791 |

### Distribucion de severidad en el dataset

| Nivel | Horas | % del total |
|---|---|---|
| Verde_Amarillo | 817 | 4.0% |
| Presion_Moderada | 5,312 | 25.7% |
| Presion_Alta | 9,634 | 46.6% |
| Critico | 4,377 | 21.2% |
| Colapso | 527 | 2.5% |

### Interpretacion

- **+1h es altamente predictivo** (BA=0.88) — ideal para alertas inmediatas
- **+3h es el punto optimo** entre precision y utilidad operativa — recomendado para planificacion
- **+6h tiene rendimiento aceptable** pero con mayor incertidumbre — usar como referencia, no como decision unica
- El **MASE > 1.0** en todos los horizontes indica que la saturacion tiene una fuerte componente autoregresiva que el modelo naive (persistencia) captura parcialmente, pero nuestros modelos agregan valor predictivo real

---

## 11. Limitaciones conocidas

### Desbalance de clases
La clase Verde_Amarillo representa solo el 4% del dataset. La prediccion de niveles bajos de saturacion es inherentemente dificil.

### Rendimiento en +6h
El MASE de 2.63 sugiere que la predictibilidad se degrada significativamente mas alla de 3 horas. Considerar +6h como tendencia, no como prediccion exacta.

### Sin datos en tiempo real
El dashboard usa datos historicos simulados. La conexion a datos en tiempo real requiere integracion con el sistema de informacion hospitalaria.

### Sin memoria entre sesiones
Cada prediccion parte de los ultimos valores conocidos. No hay acumulacion de historial entre ejecuciones.

### Variables externas no incluidas
No se incorporan clima, eventos locales (conciertos, partidos), ni picos epidemiologicos (gripe, dengue).

### Sin validacion clinica
El modelo fue entrenado con datos historicos reales pero no ha sido revisado por coordinadores medicos ni validado en operacion paralela.

---

## 12. Licencia

**Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)**

Copyright (c) 2026

Uso permitido unicamente para fines academicos y no comerciales.

- **Compartir** — copiar y redistribuir el material en cualquier medio o formato
- **Adaptar** — remezclar, transformar y construir sobre el material

**Atribucion requerida** — credito apropiado, enlace a la licencia, indicacion de cambios.
**Uso no comercial** — no se puede usar con fines comerciales.


