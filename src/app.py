# =============================================================================
# Dashboard de Saturación — Urgencias
# Modelo predictivo Alpha · Datos simulados M/M/c
#
# Librerías requeridas (instalar en tu .venv):
#   pip install streamlit pandas numpy scikit-learn matplotlib plotly pickle5
#
# Para ejecutar:
#   streamlit run app.py
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pickle
from datetime import datetime, timedelta
import os

# -----------------------------------------------------------------------------
# Configuración de página — debe ser lo primero que se llama en Streamlit
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Urgencias · Panel del Coordinador",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -----------------------------------------------------------------------------
# Estilos CSS personalizados
# Usamos st.markdown con unsafe_allow_html para inyectar CSS que Streamlit
# no expone nativamente (colores semáforo, tarjetas, etc.)
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* Reducir padding por defecto de Streamlit */
    .block-container { padding-top: 1rem; padding-bottom: 1rem; }

    /* Tarjeta métrica personalizada */
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 14px 18px;
        border: 1px solid #e9ecef;
    }
    .metric-label {
        font-size: 12px;
        color: #6c757d;
        margin-bottom: 4px;
        font-weight: 500;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 600;
        color: #212529;
        line-height: 1.1;
    }
    .metric-delta {
        font-size: 11px;
        color: #6c757d;
        margin-top: 3px;
    }

    /* Semáforo */
    .semaforo-verde  { background:#d4edda; color:#155724; border:2px solid #28a745; }
    .semaforo-amarillo { background:#fff3cd; color:#856404; border:2px solid #ffc107; }
    .semaforo-rojo   { background:#f8d7da; color:#721c24; border:2px solid #dc3545; }
    .semaforo-box {
        border-radius: 50%;
        width: 120px;
        height: 120px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 16px;
        font-weight: 700;
        margin: 0 auto;
        text-align: center;
    }

    /* Badge de nivel */
    .badge-verde    { background:#d4edda; color:#155724; padding:3px 10px;
                      border-radius:12px; font-size:11px; font-weight:600; }
    .badge-amarillo { background:#fff3cd; color:#856404; padding:3px 10px;
                      border-radius:12px; font-size:11px; font-weight:600; }
    .badge-rojo     { background:#f8d7da; color:#721c24; padding:3px 10px;
                      border-radius:12px; font-size:11px; font-weight:600; }

    /* KPI row */
    .kpi-ok   { background:#d4edda; color:#155724; padding:2px 8px;
                border-radius:8px; font-size:11px; }
    .kpi-warn { background:#fff3cd; color:#856404; padding:2px 8px;
                border-radius:8px; font-size:11px; }
    .kpi-bad  { background:#f8d7da; color:#721c24; padding:2px 8px;
                border-radius:8px; font-size:11px; }

    /* Header barra superior */
    .topbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 10px 0 16px 0;
        border-bottom: 1px solid #e9ecef;
        margin-bottom: 16px;
    }
    .live-badge {
        background: #d4edda;
        color: #155724;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# FUNCIONES DE CARGA
# Se usan @st.cache_data para que Streamlit no recargue datos/modelo en cada
# interacción del usuario — mejora el rendimiento significativamente
# =============================================================================

# 1. Definición de rutas (Usando raw strings 'r' para evitar problemas con \)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RUTA_DATOS = os.path.join(BASE_DIR, "data", "urgencias_v2_colas.csv")
RUTA_MODELO = os.path.join(BASE_DIR, "models", "modelo_rf.pkl")
RUTA_FEATURES = os.path.join(BASE_DIR, "models", "features_lista.pkl")

st.write("Ruta de datos:", RUTA_DATOS)
st.write("Existe archivo:", os.path.exists(RUTA_DATOS))

@st.cache_data
def cargar_datos(path: str) -> pd.DataFrame:
    """Carga el dataset simulado y construye la columna datetime."""
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(
        df["fecha"].astype(str) + " " + df["hora"].astype(str) + ":00"
    )
    df = df.sort_values("datetime").reset_index(drop=True)
    return df



@st.cache_resource
def cargar_modelo(path: str):
    """Carga el modelo Random Forest entrenado desde el archivo .pkl."""
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_data
def cargar_features(path: str) -> list:
    """Carga la lista de features en el orden exacto que espera el modelo."""
    with open(path, "rb") as f:
        return pickle.load(f)


# -----------------------------------------------------------------------------
# Lista de features (debe coincidir exactamente con el orden del entrenamiento)
# Si no tienes features_lista.pkl, se usa esta lista por defecto
# -----------------------------------------------------------------------------
FEATURES_DEFAULT = [
    "cant_triage", "cola_esperando_inicio_hora", "camas_ocupadas",
    "abandonos_triage", "capacidad_hora",
    "cant_triage_lag1h", "cant_triage_lag2h", "cant_triage_lag3h",
    "saturacion_lag1h", "saturacion_lag3h", "saturacion_lag6h",
    "abandonos_lag1h", "cola_lag1h", "camas_ocupadas_lag1h",
    "tiempo_turnero_min", "pct_cumpl_oportunidad_triage",
    "pct_cumpl_consulta_t2", "tiempo_consulta_min",
    "hora", "dia_semana_enc", "es_festivo",
]

# Mapeo día de semana → encoding ordinal por carga (igual que en el notebook)
DIA_ENC = {
    "Monday": 6, "Tuesday": 5, "Wednesday": 4, "Thursday": 3,
    "Friday": 2, "Saturday": 1, "Sunday": 0,
}

# Umbrales del semáforo
UMBRAL_VERDE    = 0.7
UMBRAL_AMARILLO = 1.0


# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================

def nivel_semaforo(idx: float) -> str:
    """Convierte el índice numérico en nivel semáforo."""
    if idx < UMBRAL_VERDE:
        return "Verde"
    elif idx < UMBRAL_AMARILLO:
        return "Amarillo"
    return "Rojo"


def clase_semaforo(nivel: str) -> str:
    """Devuelve la clase CSS correspondiente al nivel."""
    return {
        "Verde": "semaforo-verde",
        "Amarillo": "semaforo-amarillo",
        "Rojo": "semaforo-rojo",
    }.get(nivel, "semaforo-verde")


def badge_html(nivel: str) -> str:
    """Genera el HTML del badge de nivel."""
    cls = {"Verde": "badge-verde", "Amarillo": "badge-amarillo", "Rojo": "badge-rojo"}
    return f'<span class="{cls[nivel]}">{nivel}</span>'


def kpi_status_html(valor: float, meta: float, invertido: bool = False) -> str:
    """
    Genera badge de estado para KPI.
    invertido=True cuando menor valor es mejor (ej: abandonos, estancia).
    """
    cumple = valor >= meta if not invertido else valor <= meta
    cerca  = (valor >= meta * 0.85) if not invertido else (valor <= meta * 1.15)
    if cumple:
        return '<span class="kpi-ok">OK</span>'
    elif cerca:
        return '<span class="kpi-warn">En riesgo</span>'
    return '<span class="kpi-bad">Bajo meta</span>'


def predecir_proximas_horas(modelo, df: pd.DataFrame, features: list,
                             n_horas: int = 6) -> pd.DataFrame:
    """
    Genera predicciones para las próximas n_horas usando la última fila del
    dataset como punto de partida. En producción, estos valores vendrían del
    HIS en tiempo real.

    Estrategia: toma la última observación disponible y proyecta hacia adelante
    aplicando el perfil horario promedio del dataset como guía de tendencia,
    más ruido aleatorio controlado (±4%).
    """
    ultima = df.iloc[-1].copy()
    hora_base = int(ultima["hora"])
    fecha_base = pd.to_datetime(ultima["datetime"])

    # Perfil horario promedio del dataset (normalizado)
    perfil = df.groupby("hora")["saturacion_idx"].mean()

    resultados = []
    sat_prev = float(ultima["saturacion_idx"])

    for h in range(1, n_horas + 1):
        hora_pred = (hora_base + h) % 24
        dt_pred   = fecha_base + timedelta(hours=h)

        # Construir fila de features para predecir
        fila = ultima.copy()
        fila["hora"]             = hora_pred
        fila["dia_semana_enc"]   = DIA_ENC.get(dt_pred.strftime("%A"), 3)
        fila["es_festivo"]       = 0

        # Actualizar lags con los valores simulados de la hora previa
        fila["saturacion_lag1h"] = sat_prev
        fila["cant_triage_lag1h"] = ultima["cant_triage"] * (
            perfil.get(hora_pred, 1) / max(perfil.get(hora_base, 1), 0.01)
        )

        # Predicción del modelo
        X = pd.DataFrame([fila[features]])
        pred = float(modelo.predict(X)[0])

        # Ruido leve para que la curva no sea perfectamente suave
        pred = pred * np.random.uniform(0.96, 1.04)
        pred = round(max(0.0, min(pred, 2.5)), 4)

        nivel = nivel_semaforo(pred)
        resultados.append({
            "hora_label": dt_pred.strftime("%H:%M"),
            "datetime":   dt_pred,
            "saturacion_pred": pred,
            "nivel": nivel,
        })
        sat_prev = pred

    return pd.DataFrame(resultados)


# =============================================================================
# CARGA DE DATOS Y MODELO
# Streamlit re-ejecuta el script completo en cada interacción; el cache
# garantiza que estas cargas pesadas solo ocurran una vez.
# =============================================================================

# Rutas — ajusta si mueves los archivos
df = cargar_datos(RUTA_DATOS)
modelo = cargar_modelo(RUTA_MODELO)
features = cargar_features(RUTA_FEATURES)

try:
    df = cargar_datos(RUTA_DATOS)
except FileNotFoundError:
    st.error(f"No se encontró el archivo de datos: {RUTA_DATOS}")
    st.stop()

try:
    modelo = cargar_modelo(RUTA_MODELO)
except FileNotFoundError:
    st.error(f"No se encontró el modelo: {RUTA_MODELO}")
    st.stop()

try:
    features = cargar_features(RUTA_FEATURES)
except FileNotFoundError:
    features = FEATURES_DEFAULT

# Agregar encoding de día de semana si no existe
if "dia_semana_enc" not in df.columns:
    df["dia_semana_enc"] = df["dia_semana"].map(DIA_ENC)


# =============================================================================
# SIDEBAR — Selector de fecha/hora para simular diferentes momentos del turno
# Permite al coordinador explorar predicciones en cualquier punto del histórico
# =============================================================================

with st.sidebar:
    st.markdown("### Simulador de turno")
    st.caption("Selecciona un momento del histórico para ver cómo se hubiera visto el dashboard.")

    fecha_min = df["datetime"].min().date()
    fecha_max = df["datetime"].max().date()

    fecha_sel = st.date_input(
        "Fecha",
        value=fecha_max - timedelta(days=5),
        min_value=fecha_min,
        max_value=fecha_max,
    )
    hora_sel = st.slider("Hora del turno", min_value=0, max_value=23, value=14)

    st.divider()
    st.markdown("### Parámetros del semáforo")
    umbral_verde = st.slider(
        "Umbral Verde → Amarillo", 0.3, 0.9, UMBRAL_VERDE, step=0.05
    )
    umbral_rojo = st.slider(
        "Umbral Amarillo → Rojo", 0.7, 1.5, UMBRAL_AMARILLO, step=0.05
    )

    st.divider()
    st.caption("Alpha v0.1 · Datos simulados · Modelo Random Forest")


# =============================================================================
# FILTRAR DATOS AL MOMENTO SELECCIONADO
# =============================================================================

dt_sel = pd.Timestamp(fecha_sel) + pd.Timedelta(hours=hora_sel)

# Filas hasta el momento seleccionado (simula "datos hasta ahora")
df_hasta = df[df["datetime"] <= dt_sel].copy()

if len(df_hasta) < 10:
    st.warning("No hay suficientes datos en el período seleccionado. Ajusta la fecha.")
    st.stop()

fila_actual = df_hasta.iloc[-1]

# Últimas 12 horas para la gráfica histórica
df_12h = df_hasta.tail(12).copy()

# Saturación actual
sat_actual = float(fila_actual["saturacion_idx"])
nivel_actual = nivel_semaforo(sat_actual)


# =============================================================================
# BARRA SUPERIOR — Título + estado live
# =============================================================================

col_titulo, col_badge = st.columns([3, 1])
with col_titulo:
    st.markdown(
        f"### 🏥 Urgencias — Panel del coordinador &nbsp;&nbsp;"
        f"<small style='color:#6c757d;font-weight:400'>"
        f"Turno · {dt_sel.strftime('%A %d %b · %H:%M h')}</small>",
        unsafe_allow_html=True,
    )
with col_badge:
    st.markdown(
        '<div style="text-align:right;padding-top:8px">'
        '<span class="live-badge">● Simulación activa</span></div>',
        unsafe_allow_html=True,
    )

st.divider()


# =============================================================================
# FILA 1 — Métricas operativas clave (4 tarjetas)
# Son los números que el coordinador necesita ver de un vistazo
# =============================================================================

c1, c2, c3, c4 = st.columns(4)

cola_actual    = int(fila_actual["cola_esperando_inicio_hora"])
camas_actuales = int(fila_actual["camas_ocupadas"])
triage_actual  = int(fila_actual["cant_triage"])
abandonos_act  = int(fila_actual["abandonos_triage"])

# Deltas vs hora anterior
if len(df_hasta) >= 2:
    prev = df_hasta.iloc[-2]
    delta_cola     = cola_actual - int(prev["cola_esperando_inicio_hora"])
    delta_camas    = camas_actuales - int(prev["camas_ocupadas"])
    delta_triage   = triage_actual - int(prev["cant_triage"])
    delta_abandonos = abandonos_act - int(prev["abandonos_triage"])
else:
    delta_cola = delta_camas = delta_triage = delta_abandonos = 0

with c1:
    st.metric(
        label="Pacientes en cola",
        value=cola_actual,
        delta=delta_cola,
        delta_color="inverse",  # rojo si sube (malo), verde si baja
    )

with c2:
    st.metric(
        label=f"Camas ocupadas / 30",
        value=f"{camas_actuales}",
        delta=delta_camas,
        delta_color="inverse",
    )

with c3:
    st.metric(
        label="Triage última hora",
        value=f"{triage_actual} pac",
        delta=delta_triage,
        delta_color="inverse",
    )

with c4:
    st.metric(
        label="Abandonos",
        value=abandonos_act,
        delta=delta_abandonos,
        delta_color="inverse",
    )


# =============================================================================
# FILA 2 — Semáforo + Gráfica histórica + predicción
# =============================================================================

col_sem, col_graf = st.columns([1, 3])

# --- Semáforo actual ---
with col_sem:
    cls_sem = clase_semaforo(nivel_actual)
    accion = {
        "Verde":    "Operación normal",
        "Amarillo": "Revisar altas pendientes",
        "Rojo":     "Activar protocolo de contingencia",
    }[nivel_actual]

    st.markdown(
        f"""
        <div style="text-align:center;padding:16px 8px">
          <div class="semaforo-box {cls_sem}">{nivel_actual.upper()}</div>
          <div style="margin-top:12px;font-size:14px;font-weight:600">
            Índice: {sat_actual:.2f}
          </div>
          <div style="margin-top:6px;font-size:12px;color:#6c757d;line-height:1.4">
            {accion}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# --- Gráfica histórica + predicción ---
with col_graf:
    # Generar predicción
    df_pred = predecir_proximas_horas(modelo, df_hasta, features, n_horas=6)

    fig = go.Figure()

    # Bandas de color semáforo (fondo)
    for y0, y1, color in [
        (0, umbral_verde, "rgba(40,167,69,0.07)"),
        (umbral_verde, umbral_rojo, "rgba(255,193,7,0.10)"),
        (umbral_rojo, 2.5, "rgba(220,53,69,0.08)"),
    ]:
        fig.add_hrect(y0=y0, y1=y1, fillcolor=color, line_width=0)

    # Líneas de umbral
    fig.add_hline(y=umbral_verde,  line_dash="dot", line_color="#28a745",
                  line_width=1, opacity=0.6)
    fig.add_hline(y=umbral_rojo,   line_dash="dot", line_color="#dc3545",
                  line_width=1, opacity=0.6)

    # Serie histórica (últimas 12 horas)
    fig.add_trace(go.Scatter(
        x=df_12h["datetime"], y=df_12h["saturacion_idx"],
        name="Real (últimas 12h)",
        line=dict(color="#212529", width=2),
        mode="lines",
    ))

    # Punto actual
    fig.add_trace(go.Scatter(
        x=[fila_actual["datetime"]], y=[sat_actual],
        name="Ahora",
        mode="markers",
        marker=dict(size=10, color="#dc3545", symbol="circle"),
        showlegend=True,
    ))

    # Predicción (línea punteada)
    # Unir con el punto actual para que la curva no quede flotando
    x_pred = [fila_actual["datetime"]] + list(df_pred["datetime"])
    y_pred = [sat_actual] + list(df_pred["saturacion_pred"])

    fig.add_trace(go.Scatter(
        x=x_pred, y=y_pred,
        name="Predicción (6h)",
        line=dict(color="#007bff", width=2, dash="dash"),
        mode="lines",
    ))

    fig.update_layout(
        title=dict(text="Saturación — últimas 12h + predicción 6h",
                   font=dict(size=13), x=0),
        height=220,
        margin=dict(l=0, r=0, t=32, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1, font=dict(size=11)),
        xaxis=dict(showgrid=False, tickformat="%H:%M"),
        yaxis=dict(showgrid=True, gridcolor="#f0f0f0",
                   title="Índice", range=[0, 2.0]),
        plot_bgcolor="white",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# =============================================================================
# FILA 3 — Tabla predicción 6h + KPIs de calidad
# =============================================================================

col_pred, col_kpi = st.columns(2)

# --- Tabla de predicción por hora ---
with col_pred:
    st.markdown("**Predicción próximas 6 horas**")

    # Construir HTML de la tabla manualmente para aplicar colores por nivel
    filas_html = ""
    for _, row in df_pred.iterrows():
        badge = badge_html(row["nivel"])
        # Barra de progreso proporcional al índice (máx 2.0 = 100%)
        pct = min(int(row["saturacion_pred"] / 2.0 * 100), 100)
        color_barra = {
            "Verde": "#28a745",
            "Amarillo": "#ffc107",
            "Rojo": "#dc3545",
        }[row["nivel"]]
        filas_html += f"""
        <tr style="border-bottom:1px solid #f0f0f0">
          <td style="padding:7px 8px;font-size:13px;color:#495057;width:55px">
            {row['hora_label']}
          </td>
          <td style="padding:7px 8px;width:120px">
            <div style="background:#f0f0f0;border-radius:4px;height:8px;overflow:hidden">
              <div style="width:{pct}%;height:100%;background:{color_barra};border-radius:4px"></div>
            </div>
          </td>
          <td style="padding:7px 8px;font-size:13px;font-weight:500;width:45px">
            {row['saturacion_pred']:.2f}
          </td>
          <td style="padding:7px 8px">{badge}</td>
        </tr>
        """

    st.markdown(
        f"""
        <table style="width:100%;border-collapse:collapse">
          <thead>
            <tr style="border-bottom:2px solid #dee2e6">
              <th style="text-align:left;padding:6px 8px;font-size:11px;
                         color:#6c757d;font-weight:500">Hora</th>
              <th style="text-align:left;padding:6px 8px;font-size:11px;
                         color:#6c757d;font-weight:500">Nivel</th>
              <th style="text-align:left;padding:6px 8px;font-size:11px;
                         color:#6c757d;font-weight:500">Índice</th>
              <th style="text-align:left;padding:6px 8px;font-size:11px;
                         color:#6c757d;font-weight:500">Estado</th>
            </tr>
          </thead>
          <tbody>{filas_html}</tbody>
        </table>
        """,
        unsafe_allow_html=True,
    )

# --- KPIs de calidad (Resolución 256) ---
with col_kpi:
    st.markdown("**Indicadores de calidad — turno actual**")

    # Calcular promedios del turno (últimas 8 horas)
    df_turno = df_hasta.tail(8)

    kpis = [
        {
            "nombre": "Oportunidad triage",
            "valor": df_turno["pct_cumpl_oportunidad_triage"].mean(),
            "meta": 0.85,
            "formato": "{:.0%}",
            "invertido": False,
        },
        {
            "nombre": "Cumplimiento T2",
            "valor": df_turno["pct_cumpl_consulta_t2"].mean(),
            "meta": 0.90,
            "formato": "{:.0%}",
            "invertido": False,
        },
        {
            "nombre": "Cumplimiento T3",
            "valor": df_turno["pct_cumpl_consulta_t3"].mean(),
            "meta": 0.80,
            "formato": "{:.0%}",
            "invertido": False,
        },
        {
            "nombre": "Estancia prolongada",
            "valor": df_turno["pct_estancia_prolongada"].mean(),
            "meta": 0.20,
            "formato": "{:.0%}",
            "invertido": True,   # menor es mejor
        },
        {
            "nombre": "Reingresos 72h",
            "valor": df_turno["pct_reingresos_cie10"].mean(),
            "meta": 0.05,
            "formato": "{:.1%}",
            "invertido": True,
        },
        {
            "nombre": "T. prom. consulta",
            "valor": df_turno["tiempo_consulta_min"].mean(),
            "meta": 60,
            "formato": "{:.0f} min",
            "invertido": True,
        },
    ]

    filas_kpi = ""
    for kpi in kpis:
        val_fmt  = kpi["formato"].format(kpi["valor"])
        badge    = kpi_status_html(kpi["valor"], kpi["meta"], kpi["invertido"])
        filas_kpi += f"""
        <tr style="border-bottom:1px solid #f0f0f0">
          <td style="padding:7px 8px;font-size:13px;color:#495057">{kpi['nombre']}</td>
          <td style="padding:7px 8px;font-size:13px;font-weight:500">{val_fmt}</td>
          <td style="padding:7px 8px">{badge}</td>
        </tr>
        """

    st.markdown(
        f"""
        <table style="width:100%;border-collapse:collapse">
          <thead>
            <tr style="border-bottom:2px solid #dee2e6">
              <th style="text-align:left;padding:6px 8px;font-size:11px;
                         color:#6c757d;font-weight:500">Indicador</th>
              <th style="text-align:left;padding:6px 8px;font-size:11px;
                         color:#6c757d;font-weight:500">Valor</th>
              <th style="text-align:left;padding:6px 8px;font-size:11px;
                         color:#6c757d;font-weight:500">Estado</th>
            </tr>
          </thead>
          <tbody>{filas_kpi}</tbody>
        </table>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# FILA 4 — Análisis histórico (heatmap + distribución semanal)
# Permite al coordinador entender los patrones estructurales del servicio
# =============================================================================

st.divider()
st.markdown("**Análisis histórico de patrones**")

col_heat, col_sem_hist = st.columns(2)

with col_heat:
    # Heatmap hora × día de semana
    orden_dias = ["Monday","Tuesday","Wednesday","Thursday",
                  "Friday","Saturday","Sunday"]
    etiquetas  = ["Lun","Mar","Mié","Jue","Vie","Sáb","Dom"]

    pivot = df.pivot_table(
        values="saturacion_idx",
        index="hora",
        columns="dia_semana",
        aggfunc="mean",
    ).reindex(columns=orden_dias)
    pivot.columns = etiquetas

    fig_heat = px.imshow(
        pivot,
        color_continuous_scale="RdYlGn_r",
        aspect="auto",
        zmin=0.3, zmax=1.5,
        labels=dict(x="Día", y="Hora", color="Saturación"),
        title="Mapa de calor — saturación promedio por hora y día",
    )
    fig_heat.update_layout(
        height=300,
        margin=dict(l=0, r=0, t=36, b=0),
        coloraxis_colorbar=dict(thickness=12, len=0.8),
        title=dict(font=dict(size=13)),
    )
    st.plotly_chart(fig_heat, use_container_width=True,
                    config={"displayModeBar": False})

with col_sem_hist:
    # Distribución del semáforo por día de semana
    dist = df.groupby(["dia_semana", "nivel_saturacion"]).size().reset_index(name="n")
    dist["dia_semana"] = pd.Categorical(dist["dia_semana"],
                                        categories=orden_dias, ordered=True)
    dist = dist.sort_values("dia_semana")
    dist["dia_label"] = dist["dia_semana"].map(
        dict(zip(orden_dias, etiquetas))
    )

    color_map = {"Verde": "#28a745", "Amarillo": "#ffc107", "Rojo": "#dc3545"}
    fig_bar = px.bar(
        dist,
        x="dia_label", y="n", color="nivel_saturacion",
        color_discrete_map=color_map,
        title="Distribución semáforo por día de semana",
        labels={"n": "Horas", "dia_label": "", "nivel_saturacion": "Nivel"},
        barmode="stack",
    )
    fig_bar.update_layout(
        height=300,
        margin=dict(l=0, r=0, t=36, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1, font=dict(size=11)),
        plot_bgcolor="white",
        paper_bgcolor="rgba(0,0,0,0)",
        title=dict(font=dict(size=13)),
        yaxis=dict(showgrid=True, gridcolor="#f0f0f0"),
        xaxis=dict(showgrid=False),
    )
    st.plotly_chart(fig_bar, use_container_width=True,
                    config={"displayModeBar": False})


# =============================================================================
# FOOTER
# =============================================================================

st.divider()
st.markdown(
    """
    <div style="text-align:center;font-size:11px;color:#adb5bd;padding:8px 0">
        Alpha v0.1 · Datos simulados (180 días · modelo M/M/c) ·
        Modelo Random Forest · No usar en entorno clínico real sin validación
    </div>
    """,
    unsafe_allow_html=True,
)