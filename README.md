# Sistema Predictivo de Saturación en Urgencias

> **Alpha v0.1** — Prototipo funcional con datos simulados  

---

## Tabla de contenidos

1. [La problemática](#1-la-problemática)
2. [Nuestra solución](#2-nuestra-solución)
3. [Estado del proyecto: Alpha](#3-estado-del-proyecto-alpha)
4. [Los datos simulados](#4-los-datos-simulados)
5. [Arquitectura del sistema](#5-arquitectura-del-sistema)
6. [Estructura del repositorio](#6-estructura-del-repositorio)
7. [Requisitos](#7-requisitos)
8. [Instalación y ejecución paso a paso](#8-instalación-y-ejecución-paso-a-paso)
9. [Uso del dashboard](#9-uso-del-dashboard)
10. [Modelo predictivo](#10-modelo-predictivo)
11. [Limitaciones conocidas](#11-limitaciones-conocidas)
12. [Hoja de ruta](#12-hoja-de-ruta)
13. [Equipo](#13-equipo)

---

## 1. La problemática

Los servicios de urgencias en Colombia operan frecuentemente por encima del **120% de su capacidad instalada**. Esta saturación no es un evento puntual, sino un estado crónico que genera consecuencias clínicas, operativas y legales:

- **Demoras en triage** que violan los tiempos establecidos por la Resolución 256 de 2016 del Ministerio de Salud
- **Abandono de pacientes** antes de ser atendidos — un indicador directo de colapso del sistema
- **Fatiga y error humano** en el personal médico y de enfermería
- **Riesgos legales** para la IPS por incumplimiento de estándares de calidad

Los coordinadores médicos y jefes de enfermería enfrentan lo que ellos mismos describen como **"ceguera operativa"**: gestionan el censo de camas minuto a minuto de forma completamente reactiva, sin poder anticipar los picos de demanda antes de que el servicio colapse físicamente.

Los sistemas actuales (HIS, HOSVITAL, reportes de Power BI) solo muestran **lo que ya pasó** — son herramientas de retrospección, no de anticipación.

### El costo real de no predecir

| Consecuencia | Impacto |
|---|---|
| Abandono antes de atención | Riesgo clínico + métrica penalizada ante MinSalud |
| Estancia prolongada en observación | Bloquea camas → retroalimenta la saturación |
| Incumplimiento oportunidad triage | Sanción Resolución 256 de 2016 |
| Reingresos en 72h | Indicador de mala calidad de atención |

---

## 2. Nuestra solución

Implementamos un **modelo de Machine Learning que analiza el estado en tiempo real del servicio** (cola de espera, camas ocupadas, tasa de llegadas, abandonos) para predecir el nivel de saturación de las **próximas 6 horas**, con tres categorías accionables:

| Semáforo | Índice | Acción recomendada |
|---|---|---|
| 🟢 Verde | < 0.70 | Operación normal |
| 🟡 Amarillo | 0.70 – 1.00 | Alertar equipo, revisar altas pendientes, anticipar refuerzos |
| 🔴 Rojo | ≥ 1.00 | Activar protocolo de contingencia |

El sistema se presenta al coordinador como un **dashboard de una sola pantalla** que combina:

- Estado operativo en tiempo real (cola, camas, triage, abandonos)
- Semáforo del momento actual
- Predicción horaria de las próximas 6 horas
- Gráfica histórica de las últimas 12 horas
- Indicadores de calidad alineados con la Resolución 256

La propuesta diferencial frente a los reportes tradicionales es que **el coordinador puede tomar decisiones de personal y de gestión de altas antes de que el colapso ocurra**, no después.

---

## 3. Estado del proyecto: Alpha

> ⚠️ Este es un prototipo en fase **Alpha**. No está diseñado para uso clínico real en esta etapa.

### ¿Por qué Alpha y por qué datos simulados?

Una fase Alpha tiene un objetivo específico: **demostrar que el modelo puede predecir el comportamiento del sistema con precisión aceptable**, antes de invertir en la integración con infraestructura hospitalaria real.

Trabajar con datos reales de pacientes en esta etapa implicaría:

1. Gestionar permisos de acceso al HIS de la IPS (proceso de semanas)
2. Construir el pipeline de anonimización antes de poder usar los datos
3. Esperar meses para acumular suficiente historia

Los datos simulados nos permiten **validar la arquitectura y el modelo en días**, y presentar resultados concretos que justifiquen la inversión en la integración real.

### Qué valida el Alpha

- ✅ La arquitectura modular del sistema funciona
- ✅ El modelo predictivo aprende patrones de estacionalidad y colas
- ✅ El dashboard es usable para un coordinador médico
- ✅ Los KPIs de la Resolución 256 se pueden calcular y mostrar automáticamente

### Qué queda para Beta

- ⏳ Conexión con datos reales del HIS (HOSVITAL)
- ⏳ Pipeline de anonimización y cumplimiento HABEAS DATA
- ⏳ Integración con Microsoft Fabric
- ⏳ Alertas push a dispositivos móviles
- ⏳ Loop de retroalimentación del coordinador
- ⏳ Reentrenamiento automático con detección de data drift

---

## 4. Los datos simulados

Los datos se generaron con un **modelo de colas M/M/c** (Markoviano con múltiples servidores), que es el modelo matemático estándar para sistemas de atención con llegadas aleatorias y capacidad limitada.

A diferencia de datos sintéticos simples (donde cada hora se calcula de forma independiente), el modelo M/M/c **mantiene estado entre horas**: la cola de pacientes que no alcanzaron a ser atendidos en una hora pasa a la siguiente, generando el efecto de "cuello de botella" que es la característica más importante del comportamiento real de urgencias.

### Parámetros de simulación

| Parámetro | Valor | Justificación |
|---|---|---|
| Período | 180 días | Mínimo para capturar 2 ciclos semanales completos |
| Capacidad médica | 12 médicos | IPS de alta complejidad, turno típico |
| Camas observación | 30 | Dotación estándar |
| Tasa atención | 1.8 pac/médico/hora | Literatura hospitalaria colombiana |

### Estacionalidad implementada

**Ciclo diario:**
- Pico mañana: ×1.5 a las 10:00 AM
- Pico noche: ×1.5 a las 7:00 PM
- Valle nocturno: ×0.3 a las 3:00 AM

**Ciclo semanal:**
- Lunes y Martes: +20% respecto a fin de semana
- Sábado/Domingo: base reducida

**Eventos especiales:**
- Festivos colombianos: +35% de demanda

### Distribuciones estadísticas

| Variable | Distribución | Parámetros | Justificación |
|---|---|---|---|
| Llegadas triage | Poisson(λ) | λ variable por hora | Proceso de llegadas aleatorio estándar |
| Tiempo consulta | Gamma(α=2.2, β=20) | 10–180 min | Cola larga para casos complejos |
| Estancia observación | Log-Normal(μ=2.1, σ=0.8) | 1–72 horas | Mayoría sale rápido, pocos duran días |
| Cumplimientos | Beta(α, β) | Variable con carga | Concentrada cerca de la meta, degrada con saturación |

### Coherencia causal

Los datos respetan las siguientes relaciones causales del sistema real:

- **Mayor cola → mayor tiempo de espera en turnero** (no independiente)
- **Mayor tiempo de espera → mayor tasa de abandono**
- **Más llegadas de las que la capacidad puede atender → cola acumulada en la hora siguiente**
- **Saturación de camas → aumenta probabilidad de estancia prolongada**

### Índice de saturación (target del modelo)

$$\text{Saturación} = 0.45 \times \frac{\text{Total en sistema}}{\text{Capacidad hora}} + 0.35 \times \frac{\text{Camas ocupadas}}{\text{Total camas}} + 0.20 \times \min\left(\frac{\text{Cola}}{15}, 1.5\right)$$

### Descarga de datos

Los datos simulados están disponibles en este repositorio:

- **[urgencias_v2_colas.csv](./urgencias_v2_colas.csv)** — Dataset principal (180 días, 4,314 filas, 43 columnas)
- **[metricas_modelo.csv](./metricas_modelo.csv)** — Métricas de validación del modelo

Para regenerar los datos desde cero, ejecuta el script de simulación incluido en el notebook.

---

## 5. Arquitectura del sistema

```
┌─────────────────────────────────────────────────────┐
│  Capa 1 · Fuentes de datos                          │
│  HOSVITAL / HIS  ·  RIPS  ·  Variables externas    │
└──────────────────────┬──────────────────────────────┘
                       │ (Alpha: CSV simulado)
┌──────────────────────▼──────────────────────────────┐
│  Capa 2 · Preparación (pandas / Microsoft Fabric)  │
│  Anonimización  ·  Normalización temporal  ·  Lags  │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│  Capa 3 · Modelo predictivo                         │
│  Random Forest Regressor  ·  modelo_rf.pkl          │
│  Input: 21 features  ·  Output: saturacion_idx      │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│  Capa 4 · Dashboard                                 │
│  Streamlit  ·  Plotly  ·  app.py                    │
│  Semáforo  ·  Predicción 6h  ·  KPIs Res.256        │
└─────────────────────────────────────────────────────┘
```

---

## 6. Estructura del repositorio

```
urgencias-saturacion-alpha/
│
├── app.py                          # Dashboard Streamlit principal
├── modelo_saturacion_urgencias.ipynb  # Notebook de entrenamiento
│
├── data/
│   ├── urgencias_v2_colas.csv      # Dataset simulado (180 días)
│   └── metricas_modelo.csv         # Métricas de validación
│
├── models/
│   ├── modelo_rf.pkl               # Random Forest entrenado
│   ├── modelo_prophet.pkl          # Prophet (baseline)
│   └── features_lista.pkl          # Lista de features en orden
│
├── requirements.txt                # Dependencias Python
└── README.md                       # Este archivo
```

---

## 7. Requisitos

- Python 3.9 o superior
- pip
- Recomendado: entorno virtual (`.venv`)

### Dependencias

```txt
streamlit>=1.32.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
plotly>=5.18.0
prophet>=1.1.5
matplotlib>=3.7.0
seaborn>=0.12.0
```

---

## 8. Instalación y ejecución paso a paso

### Paso 1 — Clonar el repositorio

```bash
git clone https://github.com/tu-usuario/urgencias-saturacion-alpha.git
cd urgencias-saturacion-alpha
```

### Paso 2 — Crear y activar el entorno virtual

```bash
# Crear el entorno virtual
python -m venv .venv

# Activar en macOS / Linux
source .venv/bin/activate

# Activar en Windows
.venv\Scripts\activate
```

### Paso 3 — Instalar dependencias

```bash
pip install -r requirements.txt
```

> **Nota sobre Prophet:** En algunos sistemas la instalación de Prophet requiere tener instalado previamente `pystan`. Si falla, ejecuta:
> ```bash
> pip install pystan==2.19.1.1
> pip install prophet
> ```

### Paso 4 — Verificar que los archivos existen

Asegúrate de que los siguientes archivos estén en la raíz del proyecto (o en `models/` y `data/` según la estructura):

```
urgencias_v2_colas.csv
modelo_rf.pkl
features_lista.pkl
```

Si no tienes los archivos `.pkl`, puedes regenerarlos ejecutando el notebook completo:

```bash
jupyter notebook modelo_saturacion_urgencias.ipynb
# Ejecutar todas las celdas → los archivos .pkl se generan en la Celda 28
```

### Paso 5 — Ejecutar el dashboard

```bash
streamlit run app.py
```

Streamlit abrirá automáticamente el navegador en `http://localhost:8501`.

### Paso 6 — Explorar el dashboard

1. En el **sidebar izquierdo**, selecciona una fecha y hora para simular un turno
2. Ajusta los **umbrales del semáforo** si quieres probar diferentes sensibilidades
3. El panel principal mostrará el estado del sistema en ese momento y la predicción de las próximas 6 horas

---

## 9. Uso del dashboard

### Panel principal

| Sección | Qué muestra | Cuándo usarla |
|---|---|---|
| Métricas superiores | Cola, camas, triage, abandonos | Al sentarse en el turno |
| Semáforo | Estado actual del sistema | Verificación rápida |
| Gráfica 12h + predicción | Tendencia histórica y próximas 6h | Antes de cambio de guardia |
| Tabla predicción 6h | Hora a hora con nivel de alerta | Para planificar refuerzos |
| KPIs de calidad | Indicadores Resolución 256 | Reporte a coordinación médica |
| Heatmap histórico | Patrones por hora y día | Planificación de turnos |

### Sidebar — Simulador de turno

Dado que los datos son históricos simulados, el sidebar permite "viajar en el tiempo" a cualquier momento del período para ver cómo se hubiera visto el dashboard en ese instante. En producción con datos en tiempo real, este selector no sería necesario.

---

## 10. Modelo predictivo

### Algoritmo seleccionado: Random Forest Regressor

Se evaluaron dos modelos en el notebook:

| Modelo | MAE | R² | Precisión semáforo |
|---|---|---|---|
| Prophet (baseline) | 0.1513 | 0.628 | — |
| **Random Forest** | **0.0126** | **0.995** | **97.2%** |

Random Forest fue seleccionado por su capacidad de incorporar el **estado en tiempo real del sistema** (cola, camas ocupadas, abandonos) como features, lo que Prophet no puede hacer al ser un modelo puramente de serie temporal.

> ⚠️ **Nota importante sobre el R²:** El valor de 0.995 es inusualmente alto y se explica en parte por la naturaleza de los datos simulados (bajo ruido, relaciones causales perfectamente deterministas). En producción con datos reales se espera un R² entre 0.65 y 0.80. Ver la sección de [Limitaciones](#11-limitaciones-conocidas).

### Features más importantes

1. `saturacion_lag1h` — saturación de la hora anterior
2. `cola_esperando_inicio_hora` — pacientes en espera actual
3. `saturacion_lag3h` — tendencia de las últimas 3 horas
4. `cant_triage` — llegadas en la hora actual
5. `camas_ocupadas` — presión en observación

### Falsos negativos en nivel Rojo

El error más crítico clínicamente es no alertar cuando el sistema está en rojo. En la validación de 30 días:

- Casos reales Rojo: 103
- No detectados por el modelo: **5 (4.9%)**

---

## 11. Limitaciones conocidas

### R² artificialmente alto

Los datos simulados tienen muy poco ruido porque fueron generados por una fórmula. En datos reales hay eventos imprevisibles (llegada masiva de accidentados, ausencia de médico, corte de servicios) que el modelo no puede anticipar. **No interpretar el 99.5% como la precisión esperada en producción.**

### Sin acumulación de historia real

El modelo no tiene memoria entre sesiones — cada predicción parte de los últimos valores conocidos. En producción esto se resuelve con una base de datos que persiste el estado del sistema.

### Variables externas no incluidas

El Alpha no incluye clima, eventos locales (partidos de fútbol, conciertos), ni picos epidemiológicos (gripa, dengue). Estas variables pueden aumentar la demanda hasta un 40% sobre el patrón normal.

### No validado clínicamente

El modelo fue entrenado y validado con datos simulados. Antes de cualquier uso en un entorno clínico real debe pasar por:
1. Validación con datos históricos reales de la IPS
2. Revisión por coordinadores médicos y jefes de enfermería
3. Ajuste de umbrales del semáforo según criterio clínico
4. Período de operación paralela (modelo + método actual) antes de confiar en él

---

## 12. Hoja de ruta

```
Alpha (actual)         Beta                    Producción
─────────────────      ──────────────────────  ──────────────────────
✅ Datos simulados  →  ⏳ Datos reales HIS   → 🎯 Datos en tiempo real
✅ Modelo RF local  →  ⏳ Pipeline Fabric     → 🎯 Reentrenamiento auto
✅ Dashboard local  →  ⏳ Deploy cloud        → 🎯 Alertas push móvil
✅ Validación ret.  →  ⏳ Validación clínica  → 🎯 Feedback coordinador
                        ⏳ Anonimización       → 🎯 Multi-IPS (SaaS)
```

---

## 13. Equipo

Proyecto desarrollado como parte de tesis de grado en análisis de datos en salud.

---

<div align="center">
  <sub>Alpha v0.1 · No apto para uso clínico sin validación · Colombia 2025</sub>
</div>