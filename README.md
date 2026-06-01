<div align="center">

<img src="https://img.shields.io/badge/versi%C3%B3n-0.3-blue?style=for-the-badge" />
<img src="https://img.shields.io/badge/python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
<img src="https://img.shields.io/badge/streamlit-1.32+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
<img src="https://img.shields.io/badge/Docker-ready-2496ED?style=for-the-badge&logo=docker&logoColor=white" />
<img src="https://img.shields.io/badge/licencia-CC%20BY--NC%204.0-lightgrey?style=for-the-badge" />

# 🏥 Sistema Predictivo de Saturación — Urgencias

**Anticipa el colapso del servicio con 1, 3 y 6 horas de antelación**

*Datos reales · Clínica Nuestra Cali (NIT 805023423CL) · 2024 – 2026*

[Ver notebook →](notebooks/modelo_severidad_urgencias_v3.ipynb) · [Reportar un problema](../../issues) · [Hoja de ruta](#12-hoja-de-ruta)

</div>

---

## El problema en una línea

> Los coordinadores de urgencias gestionan el censo de camas **de forma reactiva**, sin poder anticipar los picos de demanda antes de que el servicio colapse.

El servicio opera crónicamente saturado — el **95% del tiempo está por encima de su capacidad instalada**. Con el semáforo tradicional de tres colores (Verde / Amarillo / Rojo), la alerta queda fija en rojo permanentemente y deja de ser útil. Este sistema reemplaza esa alerta inútil por una **predicción de severidad accionable**.

---

## Tabla de contenidos

1. [La solución](#1-la-solucion)
2. [Métricas de rendimiento](#2-metricas-de-rendimiento)
3. [Arquitectura](#3-arquitectura)
4. [Estructura del repositorio](#4-estructura-del-repositorio)
5. [Requisitos](#5-requisitos)
6. [Instalación](#6-instalacion)
7. [Uso del dashboard](#7-uso-del-dashboard)
8. [Modelo predictivo](#8-modelo-predictivo)
9. [Limitaciones conocidas](#9-limitaciones-conocidas)
10. [Producción y despliegue](#10-produccion-y-despliegue)
11. [Licencia](#11-licencia)
12. [Hoja de ruta](#12-hoja-de-ruta)

---

## 1. La solución

En lugar de preguntar *"¿está en Rojo?"* (respuesta obvia el 95% del tiempo), el sistema pregunta **"¿qué tan grave estará en las próximas horas?"**

Tres modelos Random Forest independientes analizan el estado operativo en tiempo real y predicen el **nivel de severidad** a +1h, +3h y +6h.

### Los 5 niveles de severidad operativa

> Los umbrales fueron definidos a partir de la distribución real del dataset (percentiles p4, p30, p77, p97) y validados operativamente.

| # | Nivel | Índice | % del tiempo | Acción recomendada |
|:-:|-------|:------:|:------------:|---------------------|
| 🟢 | **Operación normal** | < 1.0 | 4% | Sin intervención — mantener monitoreo |
| 🟡 | **Presión moderada** | 1.0 – 1.8 | 26% | Revisar altas pendientes · Preparar refuerzos |
| 🟠 | **Presión alta** | 1.8 – 2.8 | 47% | Activar protocolo preventivo de personal |
| 🔴 | **Crítico** | 2.8 – 3.8 | 21% | Desviar ambulancias · Movilizar guardia de reserva |
| ⚫ | **Colapso** | > 3.8 | 3% | Activar plan de crisis · Notificar dirección médica |

### ¿Por qué 5 niveles en vez de 3?

```
Semáforo anterior (3 colores)        Nuevo enfoque (5 niveles)
─────────────────────────────        ──────────────────────────
🟢 Verde    →   1%  del tiempo       🟢 Operación normal  →  4%
🟡 Amarillo →   4%  del tiempo       🟡 Presión moderada  → 26%
🔴 Rojo     →  95%  del tiempo       🟠 Presión alta      → 47%
                                     🔴 Crítico           → 21%
⚠️  Alarma fija = nadie reacciona    ⚫ Colapso           →  3%
```

Con 5 niveles todas las categorías tienen representación real, las métricas son informativas y el personal tiene **protocolos diferenciados** para cada estado.

---

## 2. Métricas de rendimiento

### Clasificación — test holdout (últimos 90 días, nunca vistos)

| Horizonte | Balanced Accuracy | F1-macro | Uso recomendado |
|:---------:|:-----------------:|:--------:|-----------------|
| **+1h** | **0.880** | **0.888** | Alertas inmediatas al coordinador |
| +3h | 0.781 | 0.818 | ⭐ Planificación del turno en curso |
| +6h | 0.745 | 0.791 | Visión estratégica — turno siguiente |

> **¿Por qué Balanced Accuracy y no accuracy?**  
> Con clases desbalanceadas, predecir siempre "Presión Alta" daría 47% de accuracy — una cifra alta pero inútil. Balanced Accuracy promedia el recall de cada clase y no se infla por la clase mayoritaria.

### Regresión — validación cruzada temporal (5 folds)

| Horizonte | MAE | RMSE | R² | MASE |
|:---------:|:---:|:----:|:--:|:----:|
| +1h | 0.161 | 0.195 | 0.880 | 1.47 |
| +3h | 0.218 | 0.270 | 0.824 | 2.07 |
| +6h | 0.269 | 0.335 | 0.754 | 2.63 |

> **Nota sobre MASE > 1.0:** la saturación hospitalaria tiene fuerte autocorrelación (el estado de ahora predice el de dentro de una hora casi perfectamente). El modelo naive de persistencia es difícil de superar en CV, pero los modelos sí aportan valor diferencial en la **clasificación por niveles**, especialmente para detectar transiciones hacia Crítico y Colapso.

---

## 3. Arquitectura

```
┌─────────────────────────────────────────────────────────────┐
│                 Power BI / Microsoft Fabric                 │
│                  Dataset de urgencias                       │
└───────────────────────────┬─────────────────────────────────┘
                            │ DAX queries
                            ▼
                   extraer_datos.ps1
                            │
           ┌────────────────┼────────────────┐
           ▼                ▼                ▼
  triage_metricas   triage_timestamps  urgencias_timestamps
  _horarias.csv         .csv               .csv
           └────────────────┬────────────────┘
                            │ ETL · sin data leakage
                            ▼
                  preparar_dataset.py
                            │
                            ▼
              dataset_train.csv  (20,667 filas · 24 cols)
                            │
               ┌────────────┴────────────┐
               ▼                         ▼
   modelo_severidad_v3.ipynb      retrain_model.py
   (exploración + validación)     (producción)
               └────────────┬────────────┘
                            │
          ┌─────────────────┼─────────────────┐
          ▼                 ▼                 ▼
  modelo_rf_1h.pkl  modelo_rf_3h.pkl  modelo_rf_6h.pkl
          │                 │                 │
          └─────────────────┼─────────────────┘
                            │
                      config_v3.pkl
                     features_v3.pkl
                            │
                            ▼
                    src/app.py  (Streamlit)
                            │
                            ▼
                 http://localhost:8501
```

### Pipeline de datos

| Paso | Script | Qué hace |
|:----:|--------|----------|
| 1 | `extraer_datos.ps1` | Consulta Power BI/Fabric con DAX y descarga 3 CSVs |
| 2 | `preparar_dataset.py` | Limpia, alinea temporalmente, genera lags sin data leakage |
| 3 | `retrain_model.py` | Entrena los 3 RF con TimeSeriesSplit y exporta artefactos |
| 4 | `app.py` | Carga modelos y sirve el dashboard en tiempo real |

---

## 4. Estructura del repositorio

```
saturacion-urgencias/
│
├── src/
│   ├── app.py                    # Dashboard Streamlit (758 líneas)
│   ├── preparar_dataset.py       # ETL pipeline
│   ├── extraer_datos.ps1         # Extracción Power BI / Fabric
│   └── queries_extraccion.md     # Documentación de queries DAX
│
├── notebooks/
│   ├── modelo_severidad_urgencias_v3.ipynb   # ← Notebook principal (v3)
│   ├── modelo_saturacion_urgencias_v2.ipynb  # Histórico v2
│   ├── modelo_saturacion_urgencias.ipynb     # Histórico v1
│   └── retrain_model.py                      # Script de reentrenamiento
│
├── data/                          # ★ No incluido en el repo (ver .gitignore)
│   └── dataset_train.csv          #   20,667 filas · generado por preparar_dataset.py
│
├── models/                        # ★ No incluido en el repo (ver .gitignore)
│   ├── modelo_rf_1h.pkl           #   ~50 MB
│   ├── modelo_rf_3h.pkl           #   ~50 MB
│   ├── modelo_rf_6h.pkl           #   ~50 MB
│   ├── config_v3.pkl              #   Umbrales, niveles, colores
│   └── features_v3.pkl            #   Lista de 24 features en orden exacto
│
├── Dockerfile                     # Imagen lista para producción
├── docker-compose.yml             # Orquestación local / servidor interno
├── pyproject.toml                 # Dependencias Python
├── uv.lock                        # Lockfile reproducible
├── .gitignore
├── LICENSE                        # CC BY-NC 4.0
└── README.md
```

> **★ Datos y modelos** no están en el repositorio por peso (~150 MB modelos + datos privados de pacientes). Al hacer `docker compose up` los modelos se entrenan automáticamente durante el build.

---

## 5. Requisitos

| Requisito | Versión | Notas |
|-----------|:-------:|-------|
| Docker | 24+ | [Instalar Docker](https://docs.docker.com/get-docker/) |
| RAM | 4 GB | 8 GB recomendado para entrenamiento |
| Disco | 2 GB | Modelos (~150 MB) + datos + imagen Docker |

### Dependencias principales

| Paquete | Versión | Uso |
|---------|:-------:|-----|
| `streamlit` | ≥ 1.32 | Dashboard web |
| `scikit-learn` | ≥ 1.3 | Random Forest + métricas |
| `pandas` | ≥ 2.0 | Manipulación de datos |
| `plotly` | ≥ 5.18 | Gráficos interactivos |
| `matplotlib` / `seaborn` | ≥ 3.7 / ≥ 0.12 | Visualización en notebook |

---

## 6. Instalación

### Una línea — Docker (recomendado)

```bash
git clone https://github.com/willyoung21/Saturacion_urgencias.git
cd Saturacion_urgencias
docker compose up
```

Esto:
1. Construye la imagen con UV y Python 3.11
2. Instala todas las dependencias
3. Entrena los 3 modelos automáticamente (2–5 min)
4. Inicia el dashboard en `http://localhost:8501`

La primera vez tomará ~5 minutos (descargar base image + instalar deps + entrenar). Las siguientes serán instantáneas gracias al caché de Docker.

### Sin Docker — desarrollo local

```bash
# 1. Clonar
git clone https://github.com/willyoung21/Saturacion_urgencias.git
cd Saturacion_urgencias

# 2. Instalar dependencias (requiere Python 3.11+ y UV)
uv sync

# 3. Entrenar modelos
uv run python notebooks/retrain_model.py

# 4. Iniciar dashboard
uv run streamlit run src/app.py
```

### Reentrenamiento con datos nuevos

```bash
# Con Docker
docker compose run dashboard uv run python notebooks/retrain_model.py

# Sin Docker
uv run python notebooks/retrain_model.py
```

---

## 7. Uso del dashboard

El dashboard está diseñado para ser operado por coordinadores médicos sin conocimiento técnico.

### Secciones del panel principal

| Sección | Qué muestra | Cuándo usarla |
|---------|-------------|---------------|
| **Banner de severidad** | Nivel actual, índice numérico, tendencia y acción recomendada | Primera vista al turno |
| **Métricas operativas** | Cola, camas, triage/hora, abandonos — con delta respecto a la hora anterior | Evaluación rápida |
| **Gráfica histórica + predicción** | Últimas 12h reales + proyección a +1h/+3h/+6h | Antes del cambio de guardia |
| **Tarjetas de predicción** | Nivel esperado por horizonte con barra de progreso y tendencia | Planificar refuerzos |
| **KPIs de calidad** | 5 indicadores con estado OK / En riesgo / Bajo | Reporte a coordinación |
| **Heatmap histórico** | Saturación promedio por hora y día de semana | Planificación de turnos |

### Sidebar — simulación histórica

Selecciona cualquier **fecha y hora** del período 2024–2026 para ver qué habría mostrado el dashboard en ese momento. Útil para revisar incidentes pasados o mostrar el sistema en reuniones.

### Interpretación de la tendencia

| Indicador | Significado | Acción sugerida |
|:---------:|-------------|-----------------|
| ⬆️ **Al alza** | La saturación aumentará en las próximas horas | Activar refuerzos preventivamente |
| ➡️ **Estable** | El sistema se mantendrá en el nivel actual | Mantener monitoreo |
| ⬇️ **A la baja** | La presión disminuirá | Relajar protocolos si aplica |

---

## 8. Modelo predictivo

### Algoritmo

**Random Forest Regressor** — tres modelos independientes, uno por horizonte.

Cada modelo predice el **índice de saturación numérico** futuro. La clasificación en niveles de severidad se aplica después, usando los umbrales definidos en la tabla de la sección 1.

```
estado actual del sistema  →  RF +1h  →  índice numérico  →  nivel de severidad
                           →  RF +3h  →  índice numérico  →  nivel de severidad
                           →  RF +6h  →  índice numérico  →  nivel de severidad
```

### Features (25 en total)

**Estado actual — señales en tiempo real (6):**
```
saturacion_idx, cant_triage, cola_esperando_inicio_hora,
camas_ocupadas, capacidad_hora, abandonos_triage
```

**Tendencia reciente — lags del pasado (9):**
```
cant_triage_lag1h/2h/3h
saturacion_lag1h/3h/6h
abandonos_lag1h, cola_lag1h, camas_ocupadas_lag1h
```

**Calidad de atención — señales adelantadas (4):**
```
tiempo_turnero_min, pct_cumpl_oportunidad_triage,
pct_cumpl_consulta_t2, tiempo_consulta_min
```

**Estacionalidad cíclica (4) + festivos (1) + presión combinada (1):**
```
hora_sin, hora_cos, dia_sin, dia_cos   ← encoding circular, evita salto 23h→0h
es_festivo
presion_sistema = cola + camas_ocupadas
```

### Validación

- **TimeSeriesSplit (5 folds):** cada fold entrena con el pasado y valida con el futuro inmediato. Nunca se usa información del futuro para entrenar.
- **Test holdout:** últimos 90 días del dataset, completamente separados durante el entrenamiento.

---

## 9. Limitaciones conocidas

| Limitación | Impacto | Mitigación posible |
|------------|:-------:|--------------------|
| `Verde_Amarillo` = 4% del dataset | Recall bajo en niveles bajos de saturación | Oversampling o ajuste de pesos de clase |
| MASE > 1.0 en CV | El naive de persistencia es difícil de superar a corto plazo | Los modelos compensan en clasificación de transiciones |
| Sin datos en tiempo real | El dashboard usa datos históricos simulados | Integración con HIS (Historia Clínica Electrónica) |
| Sin variables externas | No incluye clima, epidemias, eventos locales | Feature engineering con datos externos (SIVIGILA, eventos) |
| Sin validación clínica formal | No revisado por coordinadores en operación paralela | Piloto de 30 días en paralelo antes de producción |
| Drift del modelo | El comportamiento del sistema puede cambiar con el tiempo | Reentrenamiento mensual automatizado |

---

## 10. Producción y despliegue

El `Dockerfile` multi-etapa del repositorio produce una imagen autónoma lista para cualquier entorno.

### Opción A — Nube (AWS, Azure, GCP)

```bash
# Construir y subir a un registry
docker build -t saturacion-urgencias .
docker tag saturacion-urgencias mi-registry.azurecr.io/saturacion-urgencias
docker push mi-registry.azurecr.io/saturacion-urgencias

# Desplegar desde el registry
# AWS → ECS (Fargate)
# Azure → App Service / Container Instances
# GCP  → Cloud Run
```

Cada servicio requiere configurar:
- Puerto expuesto: `8501`
- Variables de entorno: `STREAMLIT_SERVER_MAX_UPLOAD_SIZE=200`
- Healthcheck: `GET /` en puerto 8501
- Memoria mínima: 2 GB

### Opción B — Servidor interno de la clínica

Con `docker-compose.yml` ya configurado:

```bash
docker compose up -d
```

El servicio se reinicia automáticamente si falla (`restart: unless-stopped`). Para mantenerlo actualizado:

```bash
git pull
docker compose up -d --build
```

### Opción C — Sin Docker

```bash
# systemd service unit (ejemplo)
[Service]
ExecStart=/usr/bin/uv run streamlit run /opt/saturacion-urgencias/src/app.py --server.port=8501 --server.address=0.0.0.0
WorkingDirectory=/opt/saturacion-urgencias
Restart=always
User=app
```

### Integración con datos en tiempo real

El paso pendiente más importante para producción es conectar el dashboard al sistema de información hospitalaria (HIS):

```
HIS / HL7 FHIR  →  preparar_dataset.py  →  app.py (predicción en tiempo real)
```

### Reentrenamiento automático

```bash
# Con Docker (cron mensual)
0 2 1 * * cd /opt/saturacion-urgencias && docker compose run dashboard uv run python notebooks/retrain_model.py

# Sin Docker (cron mensual)
0 2 1 * * cd /opt/saturacion-urgencias && uv run python notebooks/retrain_model.py
```

---

## 11. Licencia

**Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)**

Uso permitido únicamente para fines académicos y no comerciales.

- ✅ Compartir y redistribuir el material
- ✅ Adaptar, transformar y construir sobre el material
- ❌ Uso comercial sin autorización expresa
- ℹ️ Atribución requerida con enlace a la licencia

---

## 12. Hoja de ruta

- [ ] Piloto de validación clínica (30 días en paralelo)
- [ ] Integración con HIS en tiempo real (HL7/FHIR)
- [ ] Alertas por WhatsApp / correo cuando se predice Colapso
- [ ] Ajuste de umbrales con retroalimentación del coordinador médico
- [ ] Reentrenamiento automático mensual vía cron
- [ ] Incorporar variables externas (SIVIGILA, festivos extendidos, eventos)
- [ ] Panel de monitoreo de drift del modelo
- [ ] Internacionalización (i18n) del dashboard

---

<div align="center">

Desarrollado con datos reales de Urgencias Nuestra Cali · 2024–2026  
Clínica Nuestra Cali · NIT 805023423CL · Cali, Colombia

</div>
