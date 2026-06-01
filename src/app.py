"""
Urgencias · Panel del Coordinador v4
=====================================
Dashboard de produccion para monitoreo y prediccion
de saturacion en servicios de urgencias hospitalarias.

Estructura esperada del proyecto:
  SATURACION_URGENCIAS/
  ├── data/
  │   └── dataset_train.csv
  ├── models/
  │   ├── config_v3.pkl
  │   ├── features_v3.pkl
  │   ├── metricas_v3.csv
  │   ├── modelo_rf_1h.pkl
  │   ├── modelo_rf_3h.pkl
  │   └── modelo_rf_6h.pkl
  └── notebooks/
      └── dashboard.py   ← este archivo
"""

import os
import pickle
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACION DE PAGINA
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Urgencias · Panel del Coordinador",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# ESTILOS GLOBALES — tema oscuro de sala de control
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* ── Fuentes ────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=DM+Sans:wght@400;500;700&display=swap');

/* ── Reset y base ──────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
.main .block-container {
    padding: 0.6rem 1.2rem 1rem 1.2rem;
    max-width: 100%;
}
[data-testid="stSidebar"] {
    background: #13161f;
    border-right: 1px solid rgba(255,255,255,0.07);
}
[data-testid="stSidebar"] * { color: #c9cdd8 !important; }
[data-testid="stSidebar"] .stMarkdown hr { border-color: rgba(255,255,255,0.1); }

/* ── Banner de severidad ───────────────────────────────── */
.sev-banner {
    padding: 14px 20px;
    border-radius: 10px;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
    border: 1px solid;
}
.sev-left  { display: flex; align-items: center; gap: 14px; }
.sev-icon  { font-size: 32px; line-height: 1; }
.sev-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px; letter-spacing: 2px;
    text-transform: uppercase; margin-bottom: 2px;
}
.sev-name  { font-size: 22px; font-weight: 700; letter-spacing: -0.4px; }
.sev-idx   {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 13px; margin-top: 2px; opacity: 0.9;
}
.sev-action {
    font-size: 13px; font-weight: 600;
    padding: 10px 18px; border-radius: 8px;
    border: 1px solid; text-align: right;
    max-width: 320px; line-height: 1.45;
}

/* Paleta por nivel */
.b-verde    { background:#052e16; border-color:#166534; color:#4ade80; }
.b-amarillo { background:#1c1002; border-color:#92400e; color:#fbbf24; }
.b-naranja  { background:#1c0a02; border-color:#9a3412; color:#fb923c; }
.b-rojo     { background:#1c0202; border-color:#991b1b; color:#f87171; }
.b-purpura  { background:#1a0b2e; border-color:#6b21a8; color:#c084fc; }
.act-verde    { background:#052e16; border-color:#166534; color:#4ade80; }
.act-amarillo { background:#1c1002; border-color:#92400e; color:#fbbf24; }
.act-naranja  { background:#1c0a02; border-color:#9a3412; color:#fb923c; }
.act-rojo     { background:#1c0202; border-color:#991b1b; color:#f87171; }
.act-purpura  { background:#1a0b2e; border-color:#6b21a8; color:#c084fc; }

/* ── Encabezado del dashboard ──────────────────────────── */
.dash-header {
    display: flex; align-items: baseline;
    justify-content: space-between;
    padding: 2px 0 8px 0;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    margin-bottom: 10px;
}
.dash-title {
    font-size: 20px; font-weight: 700;
    color: #f0f2f5; letter-spacing: -0.3px;
}
.dash-sub {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px; color: #555d72;
}
.dash-live {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px; color: #16a34a;
    display: flex; align-items: center; gap: 5px;
}

/* ── Semáforo circular ─────────────────────────────────── */
.semaforo-wrap { text-align: center; padding: 8px 0 4px 0; }
.semaforo-box {
    width: 170px; height: 170px;
    border-radius: 50%;
    display: inline-flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    line-height: 1.2; border: 3px solid;
}
.semaforo-title { font-size: 13px; font-weight: 700; letter-spacing: 0.2px; }
.semaforo-idx   {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 30px; font-weight: 700; margin-top: 2px;
}
.semaforo-trend { font-size: 28px; margin-top: -2px; }
.semaforo-reco  { font-size: 11px; color: #8b92a5; margin-top: 6px; max-width: 180px; }

/* Semaforo colores */
.sem-verde    { background:#052e16; border-color:#16a34a; color:#4ade80; }
.sem-amarillo { background:#1c1002; border-color:#d97706; color:#fbbf24; }
.sem-naranja  { background:#1c0a02; border-color:#ea580c; color:#fb923c; }
.sem-rojo     { background:#1c0202; border-color:#dc2626; color:#f87171; }
.sem-purpura  { background:#1a0b2e; border-color:#9333ea; color:#c084fc; }

/* ── Métricas superiores ───────────────────────────────── */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 8px; margin-bottom: 10px;
}
.metric-card {
    background: #171b26;
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px;
    padding: 12px 14px;
}
.metric-label { font-size: 11px; color: #555d72; margin-bottom: 3px; font-weight: 500; }
.metric-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 26px; font-weight: 600; color: #f0f2f5; line-height: 1;
}
.metric-delta { font-family: 'IBM Plex Mono', monospace; font-size: 11px; margin-top: 3px; }
.delta-bad  { color: #f87171; }
.delta-ok   { color: #4ade80; }
.delta-flat { color: #555d72; }

/* ── Tarjetas de prediccion ────────────────────────────── */
.pred-card {
    background: #171b26;
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 8px; padding: 11px 12px;
    display: flex; align-items: center; gap: 10px;
    margin-bottom: 6px;
}
.pred-horiz {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px; color: #8b92a5; min-width: 28px;
}
.pred-bar-bg  { flex:1; height:6px; background:rgba(255,255,255,0.07); border-radius:3px; overflow:hidden; }
.pred-bar-fill{ height:100%; border-radius:3px; }
.pred-val {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 15px; font-weight: 600; min-width: 38px; text-align: right;
}
.pred-badge {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px; font-weight: 600;
    padding: 2px 8px; border-radius: 4px; min-width: 52px; text-align: center;
}
.pred-trend { font-size: 14px; min-width: 16px; text-align: center; }

/* ── KPIs ──────────────────────────────────────────────── */
.kpi-row {
    background: #171b26;
    border: 1px solid rgba(255,255,255,0.08);
    border-left: 3px solid;
    border-radius: 7px; padding: 9px 12px;
    display: flex; align-items: center;
    justify-content: space-between; margin-bottom: 5px;
}
.kpi-name  { font-size: 12px; color: #8b92a5; }
.kpi-val   {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 14px; font-weight: 600;
}

/* ── Etiquetas de sección ──────────────────────────────── */
.section-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px; letter-spacing: 2px;
    text-transform: uppercase; color: #555d72;
    margin-bottom: 8px; margin-top: 4px;
}
.section-title {
    font-size: 13px; font-weight: 600; color: #c9cdd8;
    margin-bottom: 8px; padding-bottom: 6px;
    border-bottom: 1px solid rgba(255,255,255,0.07);
}

/* ── Pie de pagina ─────────────────────────────────────── */
.footer {
    text-align: center; padding: 10px 0 4px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px; color: #333a4d;
    border-top: 1px solid rgba(255,255,255,0.05);
    margin-top: 10px;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# RUTAS DE ARCHIVOS
# ─────────────────────────────────────────────────────────────────────────────

# Detecta automaticamente la raiz del proyecto
# (funciona tanto desde /notebooks como desde la raiz)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(_THIS_DIR) in ("src", "notebooks"):
    BASE_DIR = os.path.dirname(_THIS_DIR)
else:
    BASE_DIR = _THIS_DIR

RUTA_DATOS     = os.path.join(BASE_DIR, "data",   "dataset_train.csv")
RUTA_CONFIG    = os.path.join(BASE_DIR, "models", "config_v3.pkl")
RUTA_FEATURES  = os.path.join(BASE_DIR, "models", "features_v3.pkl")
RUTA_METRICAS  = os.path.join(BASE_DIR, "models", "metricas_v3.csv")
RUTAS_MODELOS  = {
    "1h": os.path.join(BASE_DIR, "models", "modelo_rf_1h.pkl"),
    "3h": os.path.join(BASE_DIR, "models", "modelo_rf_3h.pkl"),
    "6h": os.path.join(BASE_DIR, "models", "modelo_rf_6h.pkl"),
}

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES Y CONFIGURACION DE NIVELES
# ─────────────────────────────────────────────────────────────────────────────
 
DIA_ENC_REV = {6:"Monday", 5:"Tuesday", 4:"Wednesday", 3:"Thursday",
               2:"Friday", 1:"Saturday", 0:"Sunday"}
DIA_LABEL   = {"Monday":"Lun","Tuesday":"Mar","Wednesday":"Mié",
               "Thursday":"Jue","Friday":"Vie","Saturday":"Sáb","Sunday":"Dom"}
ORDEN_DIAS  = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
ETIQUETAS   = ["Lun","Mar","Mié","Jue","Vie","Sáb","Dom"]
DIA_ENC_ORDEN = [6,5,4,3,2,1,0]
 
FEATURES_DEFAULT = [
    "cant_triage","cola_esperando_inicio_hora","camas_ocupadas",
    "abandonos_triage","capacidad_hora",
    "cant_triage_lag1h","cant_triage_lag2h","cant_triage_lag3h",
    "saturacion_lag1h","saturacion_lag3h","saturacion_lag6h",
    "abandonos_lag1h","cola_lag1h","camas_ocupadas_lag1h",
    "tiempo_turnero_min","pct_cumpl_oportunidad_triage",
    "pct_cumpl_consulta_t2","tiempo_consulta_min",
    "es_festivo","hora_sin","hora_cos","dia_sin","dia_cos",
    "presion_sistema",
]
 
# Configuracion por defecto (se sobreescribe con config_v3.pkl si existe)
UMBRALES_DEFAULT = [1.0, 1.8, 2.8, 3.8]
NIVELES_DEFAULT  = ["Verde_Amarillo","Presion_Moderada","Presion_Alta","Critico","Colapso"]
COLORES_DEFAULT  = {
    "Verde_Amarillo":  "#16a34a",
    "Presion_Moderada":"#d97706",
    "Presion_Alta":    "#ea580c",
    "Critico":         "#dc2626",
    "Colapso":         "#9333ea",
}
 
NIV_LABELS = {
    "Verde_Amarillo":  "Verde / Amarillo",
    "Presion_Moderada":"Presión Moderada",
    "Presion_Alta":    "Presión Alta",
    "Critico":         "Crítico",
    "Colapso":         "Colapso",
}
NIV_ICONOS = {
    "Verde_Amarillo":  "💚",
    "Presion_Moderada":"💛",
    "Presion_Alta":    "🟠",
    "Critico":         "🔴",
    "Colapso":         "🟣",
}
SEV_ACCION = {
    "Verde_Amarillo":  "Operación normal · Mantener monitoreo de rutina",
    "Presion_Moderada":"Preparar refuerzos · Revisar altas pendientes",
    "Presion_Alta":    "Activar protocolo preventivo · Alertar coordinación médica",
    "Critico":         "ACTIVAR CONTINGENCIA · Movilizar personal adicional",
    "Colapso":         "ACTIVAR COLAPSO · Plan de crisis hospitalario",
}
BANNER_CLS = {
    "Verde_Amarillo":  ("b-verde",   "act-verde",   "sem-verde"),
    "Presion_Moderada":("b-amarillo","act-amarillo","sem-amarillo"),
    "Presion_Alta":    ("b-naranja", "act-naranja", "sem-naranja"),
    "Critico":         ("b-rojo",    "act-rojo",    "sem-rojo"),
    "Colapso":         ("b-purpura", "act-purpura", "sem-purpura"),
}
COLORES_PRED_BG = {
    "Verde_Amarillo":  ("rgba(22,163,74,0.15)",  "#4ade80"),
    "Presion_Moderada":("rgba(217,119,6,0.15)",  "#fbbf24"),
    "Presion_Alta":    ("rgba(234,88,12,0.15)",  "#fb923c"),
    "Critico":         ("rgba(220,38,38,0.15)",  "#f87171"),
    "Colapso":         ("rgba(147,51,234,0.15)", "#c084fc"),
}
 
# ─────────────────────────────────────────────────────────────────────────────
# CARGA DE RECURSOS (con cache)
# ─────────────────────────────────────────────────────────────────────────────
 
@st.cache_data
def cargar_datos(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    df["dia_nombre"] = df["dia_semana_enc"].map(DIA_ENC_REV)
    df["dia_label"]  = df["dia_nombre"].map(DIA_LABEL)
    return df
 
 
@st.cache_resource
def cargar_modelo(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)
 
 
@st.cache_data
def cargar_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)
 
 
@st.cache_data
def cargar_metricas(path: str) -> pd.DataFrame:
    return pd.read_csv(path)
 
 
# ── Carga de datos ────────────────────────────────────────────────────────────
 
try:
    df = cargar_datos(RUTA_DATOS)
except FileNotFoundError:
    st.error(f"❌ No se encontró el archivo de datos en:\n`{RUTA_DATOS}`")
    st.stop()
 
try:
    modelos = {n: cargar_modelo(r) for n, r in RUTAS_MODELOS.items()}
except FileNotFoundError as e:
    st.error(f"❌ Modelo no encontrado: {e}")
    st.stop()
 
try:
    CONFIG   = cargar_pickle(RUTA_CONFIG)
    UMBRALES = CONFIG["umbrales"]
    NIVELES  = CONFIG["niveles"]
    COLORES  = CONFIG["colores"]
except (FileNotFoundError, KeyError):
    UMBRALES = UMBRALES_DEFAULT
    NIVELES  = NIVELES_DEFAULT
    COLORES  = COLORES_DEFAULT
 
try:
    features = cargar_pickle(RUTA_FEATURES)
except FileNotFoundError:
    features = FEATURES_DEFAULT
 
try:
    df_metricas = cargar_metricas(RUTA_METRICAS)
    tiene_metricas = True
except FileNotFoundError:
    tiene_metricas = False
 
# ─────────────────────────────────────────────────────────────────────────────
# FUNCIONES DE LOGICA
# ─────────────────────────────────────────────────────────────────────────────
 
def clasificar(valor: float) -> str:
    if valor < UMBRALES[0]:  return "Verde_Amarillo"
    if valor < UMBRALES[1]:  return "Presion_Moderada"
    if valor < UMBRALES[2]:  return "Presion_Alta"
    if valor < UMBRALES[3]:  return "Critico"
    return "Colapso"
 
 
def preparar_fila(fila: pd.Series, feats: list) -> pd.DataFrame:
    d = fila.to_dict()
    hora = int(fila["hora"])
    dia  = int(fila["dia_semana_enc"])
    d["hora_sin"]      = np.sin(2 * np.pi * hora / 24)
    d["hora_cos"]      = np.cos(2 * np.pi * hora / 24)
    d["dia_sin"]       = np.sin(2 * np.pi * dia / 7)
    d["dia_cos"]       = np.cos(2 * np.pi * dia / 7)
    d["presion_sistema"] = (
        fila.get("cola_esperando_inicio_hora", 0) + fila.get("camas_ocupadas", 0)
    )
    X = pd.DataFrame([d])
    return X[feats]
 
 
def predecir(modelos_dict: dict, fila: pd.Series, feats: list) -> dict:
    X = preparar_fila(fila, feats)
    return {
        nombre: round(float(max(0.0, min(modelo.predict(X)[0], 5.0))), 4)
        for nombre, modelo in modelos_dict.items()
    }
 
 
def tendencia_global(preds: dict, actual: float) -> tuple[str, str]:
    diff = preds.get("6h", actual) - actual
    if diff > 0.15:   return "↗", "Al alza"
    if diff < -0.15:  return "↘", "A la baja"
    return "→", "Estable"
 
 
def delta_html(delta: int, inv: bool = True) -> str:
    if delta == 0:
        return '<span class="metric-delta delta-flat">→ sin cambio</span>'
    bad = (delta > 0) if inv else (delta < 0)
    cls = "delta-bad" if bad else "delta-ok"
    sym = "▲" if delta > 0 else "▼"
    return f'<span class="metric-delta {cls}">{sym} {abs(delta)}</span>'
 
 
# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — Simulador de turno
# ─────────────────────────────────────────────────────────────────────────────
 
with st.sidebar:
    st.markdown("### 🏥 Simulador de Turno")
    st.caption("Navega cualquier momento del histórico para ver el estado del sistema.")
    st.divider()
 
    fecha_min = df["datetime"].min().date()
    fecha_max = df["datetime"].max().date()
 
    fecha_sel = st.date_input(
        "📅 Fecha",
        value=fecha_max - timedelta(days=5),
        min_value=fecha_min,
        max_value=fecha_max,
    )
    hora_sel = st.slider("🕐 Hora del turno", min_value=0, max_value=23, value=14, step=1,
                         format="%d:00 h")
 
    st.divider()
 
    # Métricas del modelo (si existen)
    if tiene_metricas:
        st.markdown("**Rendimiento del modelo (test)**")
        try:
            for _, row in df_metricas.iterrows():
                nombre = str(row.get("horizonte", row.get("modelo", "—")))
                rmse   = row.get("rmse", row.get("RMSE", None))
                r2     = row.get("r2",   row.get("R2",   None))
                if rmse is not None:
                    st.metric(label=f"RMSE {nombre}", value=f"{rmse:.4f}")
                if r2 is not None:
                    st.metric(label=f"R²   {nombre}", value=f"{r2:.4f}")
        except Exception:
            st.dataframe(df_metricas, use_container_width=True)
 
    st.divider()
    st.caption("v4 · 5 niveles de severidad · RF independiente por horizonte · +1h · +3h · +6h")
 
# ─────────────────────────────────────────────────────────────────────────────
# PREPARACION DE DATOS PARA EL MOMENTO SELECCIONADO
# ─────────────────────────────────────────────────────────────────────────────
 
dt_sel   = pd.Timestamp(fecha_sel) + pd.Timedelta(hours=hora_sel)
df_hasta = df[df["datetime"] <= dt_sel].copy()
 
if len(df_hasta) < 10:
    st.warning("⚠️ No hay suficientes datos en el periodo seleccionado. Ajusta la fecha.")
    st.stop()
 
fila_actual  = df_hasta.iloc[-1]
df_12h       = df_hasta.tail(12).copy()
 
sat_actual   = float(fila_actual["saturacion_idx"])
nivel_actual = clasificar(sat_actual)
prediccion   = predecir(modelos, fila_actual, features)
trend_sym, trend_lbl = tendencia_global(prediccion, sat_actual)
 
cls_banner, cls_action, cls_sem = BANNER_CLS[nivel_actual]
 
# ─────────────────────────────────────────────────────────────────────────────
# BANNER DE SEVERIDAD
# ─────────────────────────────────────────────────────────────────────────────
 
st.markdown(
    f"""
    <div class="sev-banner {cls_banner}">
      <div class="sev-left">
        <div class="sev-icon">{NIV_ICONOS[nivel_actual]}</div>
        <div>
          <div class="sev-label">Estado del Servicio</div>
          <div class="sev-name">{NIV_LABELS[nivel_actual]}</div>
          <div class="sev-idx">Índice: {sat_actual:.3f} &nbsp;·&nbsp; {trend_sym} {trend_lbl}</div>
        </div>
      </div>
      <div class="sev-action {cls_action}">{SEV_ACCION[nivel_actual]}</div>
    </div>
    """,
    unsafe_allow_html=True,
)
 
# ─────────────────────────────────────────────────────────────────────────────
# ENCABEZADO DEL DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
 
st.markdown(
    f"""
    <div class="dash-header">
      <div>
        <span class="dash-title">Urgencias · Panel del Coordinador</span>
        &nbsp;
        <span class="dash-sub">Turno · {dt_sel.strftime('%A %d %b %Y · %H:%M h')}</span>
      </div>
      <span class="dash-live">● Simulación activa</span>
    </div>
    """,
    unsafe_allow_html=True,
)
 
# ─────────────────────────────────────────────────────────────────────────────
# MÉTRICAS SUPERIORES (4 cards)
# ─────────────────────────────────────────────────────────────────────────────
 
cola_act    = int(fila_actual["cola_esperando_inicio_hora"])
camas_act   = int(fila_actual["camas_ocupadas"])
triage_act  = int(fila_actual["cant_triage"])
aband_act   = int(fila_actual["abandonos_triage"])
capacidad   = int(fila_actual["capacidad_hora"])
 
if len(df_hasta) >= 2:
    prev = df_hasta.iloc[-2]
    d_cola  = cola_act  - int(prev["cola_esperando_inicio_hora"])
    d_camas = camas_act - int(prev["camas_ocupadas"])
    d_triage= triage_act- int(prev["cant_triage"])
    d_aband = aband_act - int(prev["abandonos_triage"])
else:
    d_cola = d_camas = d_triage = d_aband = 0
 
st.markdown(
    f"""
    <div class="metric-grid">
      <div class="metric-card">
        <div class="metric-label">Pacientes en cola</div>
        <div class="metric-value">{cola_act}</div>
        {delta_html(d_cola, inv=True)}
      </div>
      <div class="metric-card">
        <div class="metric-label">Camas ocupadas / {capacidad}</div>
        <div class="metric-value">{camas_act}</div>
        {delta_html(d_camas, inv=True)}
      </div>
      <div class="metric-card">
        <div class="metric-label">Triage última hora</div>
        <div class="metric-value">{triage_act}</div>
        {delta_html(d_triage, inv=True)}
      </div>
      <div class="metric-card">
        <div class="metric-label">Abandonos</div>
        <div class="metric-value">{aband_act}</div>
        {delta_html(d_aband, inv=True)}
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
 
# ─────────────────────────────────────────────────────────────────────────────
# FILA PRINCIPAL: Semáforo | Gráfico 12h + predicción | Predicciones + KPIs
# ─────────────────────────────────────────────────────────────────────────────
 
col_sem, col_graf, col_pred = st.columns([1.1, 3.2, 1.7])
 
# ── Semáforo ──────────────────────────────────────────────────────────────────
 
with col_sem:
    icono_tr  = {"↗": "⬆", "↘": "⬇", "→": "➡"}[trend_sym]
    reco_corta = SEV_ACCION[nivel_actual].split("·")[0].strip()
 
    st.markdown(
        f"""
        <div class="semaforo-wrap">
          <div class="semaforo-box {cls_sem}">
            <div class="semaforo-title">{NIV_LABELS[nivel_actual]}</div>
            <div class="semaforo-idx">{sat_actual:.2f}</div>
            <div class="semaforo-trend">{icono_tr}</div>
          </div>
          <div class="semaforo-reco">{reco_corta}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
 
# ── Gráfico de líneas: 12h real + predicción ─────────────────────────────────
 
with col_graf:
    fig = go.Figure()
 
    # Zonas de color por nivel
    zonas = [
        (0,            UMBRALES[0], "#16a34a"),
        (UMBRALES[0],  UMBRALES[1], "#d97706"),
        (UMBRALES[1],  UMBRALES[2], "#ea580c"),
        (UMBRALES[2],  UMBRALES[3], "#dc2626"),
        (UMBRALES[3],  5.0,         "#9333ea"),
    ]
    for y0, y1, color in zonas:
        fig.add_hrect(y0=y0, y1=y1, fillcolor=color,
                      line_width=0, opacity=0.07)
 
    # Líneas de umbral
    for u, color in zip(UMBRALES, ["#16a34a","#d97706","#ea580c","#dc2626"]):
        fig.add_hline(y=u, line_dash="dot", line_color=color,
                      line_width=1, opacity=0.5)
 
    # Serie histórica
    fig.add_trace(go.Scatter(
        x=df_12h["datetime"], y=df_12h["saturacion_idx"],
        name="Real (12h)",
        line=dict(color="#e2e8f0", width=2.5),
        mode="lines",
        hovertemplate="%{x|%H:%M}<br>Saturación: %{y:.3f}<extra></extra>",
    ))
 
    # Punto actual
    fig.add_trace(go.Scatter(
        x=[fila_actual["datetime"]], y=[sat_actual],
        name="Ahora",
        mode="markers",
        marker=dict(size=13, color="#dc2626", symbol="circle",
                    line=dict(color="white", width=2)),
        hovertemplate="%{x|%H:%M}<br>Actual: %{y:.3f}<extra></extra>",
    ))
 
    # Línea de predicción
    t_base = fila_actual["datetime"]
    t_preds  = [t_base,
                t_base + timedelta(hours=1),
                t_base + timedelta(hours=3),
                t_base + timedelta(hours=6)]
    v_preds  = [sat_actual, prediccion["1h"], prediccion["3h"], prediccion["6h"]]
    niv_preds = [nivel_actual] + [clasificar(v) for v in v_preds[1:]]
 
    fig.add_trace(go.Scatter(
        x=t_preds, y=v_preds,
        name="Predicción RF",
        line=dict(color="#3b82f6", width=2.5, dash="dash"),
        mode="lines+markers",
        marker=dict(size=[11, 8, 8, 8], color="#3b82f6",
                    line=dict(color="white", width=1.5)),
        customdata=[[n.replace("_"," ")] for n in niv_preds],
        hovertemplate="%{x|%H:%M}<br>Predicción: %{y:.3f}<br>Nivel: %{customdata[0]}<extra></extra>",
    ))
 
    fig.update_layout(
        title=dict(
            text="Últimas 12 horas · Histórico real + predicción a +1h, +3h, +6h",
            font=dict(size=13, color="#8b92a5"), x=0,
        ),
        height=268,
        margin=dict(l=0, r=0, t=34, b=0),
        plot_bgcolor="#13161f",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.01,
            xanchor="right", x=1,
            font=dict(size=11, color="#8b92a5"),
            bgcolor="rgba(0,0,0,0)",
        ),
        xaxis=dict(
            showgrid=False, tickformat="%H:%M",
            color="#555d72",
        ),
        yaxis=dict(
            showgrid=True, gridcolor="rgba(255,255,255,0.05)",
            title=dict(text="Índice", font=dict(size=11, color="#555d72")),
            range=[0, 5.2],
            tickvals=[0, UMBRALES[0], UMBRALES[1], UMBRALES[2], UMBRALES[3], 5.0],
            ticktext=["0", f"VA {UMBRALES[0]}", f"PM {UMBRALES[1]}",
                      f"PA {UMBRALES[2]}", f"CR {UMBRALES[3]}", "5.0"],
            color="#555d72",
        ),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
 
# ── Predicciones por horizonte ────────────────────────────────────────────────
 
with col_pred:
    st.markdown('<div class="section-label">Predicción por horizonte</div>',
                unsafe_allow_html=True)
 
    prev_val = sat_actual
    pred_html = ""
    for horizonte in ["1h","3h","6h"]:
        v     = prediccion[horizonte]
        nivel = clasificar(v)
        bg, fg = COLORES_PRED_BG[nivel]
        pct   = min(int((v / 5.0) * 100), 100)
        diff  = v - prev_val
        tr    = "▲" if diff > 0.05 else ("▼" if diff < -0.05 else "→")
        tr_c  = "delta-bad" if diff > 0.05 else ("delta-ok" if diff < -0.05 else "delta-flat")
        lbl   = NIV_LABELS[nivel].split(" ")[0]
        bar_color = COLORES.get(nivel, "#888")
        pred_html += f"""
        <div class="pred-card">
          <div class="pred-horiz">+{horizonte}</div>
          <div class="pred-bar-bg">
            <div class="pred-bar-fill" style="width:{pct}%;background:{bar_color}"></div>
          </div>
          <div class="pred-val">{v:.2f}</div>
          <div class="pred-badge" style="background:{bg};color:{fg}">{lbl}</div>
          <span class="pred-trend {tr_c}">{tr}</span>
        </div>"""
        prev_val = v
 
    st.markdown(pred_html, unsafe_allow_html=True)
    st.markdown(
        f'<div style="text-align:right;font-family:\'IBM Plex Mono\',monospace;'
        f'font-size:11px;color:#555d72;padding:2px 4px 0 0">'
        f'Tendencia 6h: <strong style="color:#8b92a5">{trend_sym} {trend_lbl}</strong></div>',
        unsafe_allow_html=True,
    )
 
    # ── KPIs de calidad ───────────────────────────────────────────────────────
 
    st.markdown('<div class="section-label" style="margin-top:12px">Indicadores de calidad</div>',
                unsafe_allow_html=True)
 
    df_turno = df_hasta.tail(8)
 
    kpis = [
        {"nombre":"Oportunidad triage",
         "valor": df_turno["pct_cumpl_oportunidad_triage"].mean(),
         "meta":0.85, "fmt":"pct", "inv":False},
        {"nombre":"Cumplimiento T2",
         "valor": df_turno["pct_cumpl_consulta_t2"].mean(),
         "meta":0.90, "fmt":"pct", "inv":False},
        {"nombre":"Tiempo consulta",
         "valor": df_turno["tiempo_consulta_min"].mean(),
         "meta":60, "fmt":"min", "inv":True},
        {"nombre":"Espera turnero",
         "valor": df_turno["tiempo_turnero_min"].mean(),
         "meta":30, "fmt":"min", "inv":True},
        {"nombre":"Abandonos prom/h",
         "valor": df_turno["abandonos_triage"].mean(),
         "meta":2, "fmt":"num", "inv":True},
    ]
 
    kpi_html = ""
    for k in kpis:
        v    = k["valor"]
        m    = k["meta"]
        cumple = (v >= m) if not k["inv"] else (v <= m)
        cerca  = (v >= m * 0.85) if not k["inv"] else (v <= m * 1.15)
        color  = "#4ade80" if cumple else ("#fbbf24" if cerca else "#f87171")
        bcolor = color
        if k["fmt"] == "pct": val_str = f"{v:.0%}"
        elif k["fmt"] == "min": val_str = f"{v:.0f} min"
        else: val_str = f"{v:.1f}"
        kpi_html += f"""
        <div class="kpi-row" style="border-left-color:{bcolor}">
          <div class="kpi-name">{k['nombre']}</div>
          <div class="kpi-val" style="color:{color}">{val_str}</div>
        </div>"""
 
    st.markdown(kpi_html, unsafe_allow_html=True)
 
# ─────────────────────────────────────────────────────────────────────────────
# FILA INFERIOR: Heatmap | Distribución por día
# ─────────────────────────────────────────────────────────────────────────────
 
st.markdown("---")
st.markdown('<div class="section-title">📊 Análisis histórico de patrones</div>',
            unsafe_allow_html=True)
 
col_heat, col_dist = st.columns(2)
 
# ── Heatmap hora × día ────────────────────────────────────────────────────────
 
with col_heat:
    pivot = df.pivot_table(
        values="saturacion_idx",
        index="hora",
        columns="dia_semana_enc",
        aggfunc="mean",
    ).reindex(columns=DIA_ENC_ORDEN)
    pivot.columns = ETIQUETAS
 
    fig_heat = px.imshow(
        pivot,
        color_continuous_scale=[
            [0.00, "#16a34a"],
            [0.20, "#d97706"],
            [0.50, "#ea580c"],
            [0.76, "#dc2626"],
            [1.00, "#9333ea"],
        ],
        aspect="auto",
        zmin=0.5, zmax=4.0,
        labels=dict(x="Día", y="Hora", color="Saturación"),
        title="Mapa de calor — saturación promedio por hora y día",
    )
    fig_heat.update_layout(
        height=340,
        margin=dict(l=0, r=0, t=36, b=0),
        plot_bgcolor="#13161f",
        paper_bgcolor="rgba(0,0,0,0)",
        title_font=dict(size=13, color="#8b92a5"),
        coloraxis_colorbar=dict(
            thickness=12, len=0.85,
            tickfont=dict(color="#8b92a5", size=10),
            title=dict(text="", font=dict(color="#8b92a5")),
        ),
        xaxis=dict(color="#555d72"),
        yaxis=dict(color="#555d72"),
    )
    st.plotly_chart(fig_heat, use_container_width=True, config={"displayModeBar": False})
 
# ── Distribución por día ──────────────────────────────────────────────────────
 
with col_dist:
    df_temp = df.copy()
    df_temp["severidad"] = df_temp["saturacion_idx"].apply(clasificar)
    dist = (
        df_temp.groupby(["dia_label","severidad"])
        .size().reset_index(name="n")
    )
    dist["dia_label"] = pd.Categorical(
        dist["dia_label"], categories=ETIQUETAS, ordered=True
    )
    dist = dist.sort_values("dia_label")
 
    # Mapeo de colores desde el config cargado
    color_map = {
        "Verde_Amarillo":   COLORES.get("Verde_Amarillo",  "#16a34a"),
        "Presion_Moderada": COLORES.get("Presion_Moderada","#d97706"),
        "Presion_Alta":     COLORES.get("Presion_Alta",    "#ea580c"),
        "Critico":          COLORES.get("Critico",         "#dc2626"),
        "Colapso":          COLORES.get("Colapso",         "#9333ea"),
    }
 
    fig_bar = px.bar(
        dist,
        x="dia_label", y="n", color="severidad",
        color_discrete_map=color_map,
        labels={"n":"Horas","dia_label":"","severidad":"Nivel"},
        barmode="stack",
        category_orders={"severidad": NIVELES},
        title="Distribución de severidad por día de semana",
    )
    fig_bar.update_layout(
        height=370,
        margin=dict(l=0, r=0, t=36, b=60),
        plot_bgcolor="#13161f",
        paper_bgcolor="rgba(0,0,0,0)",
        title_font=dict(size=13, color="#8b92a5"),
        legend=dict(
            orientation="h", yanchor="top", y=-0.18,
            xanchor="center", x=0.5,
            font=dict(size=10, color="#8b92a5"),
            bgcolor="rgba(0,0,0,0)",
            traceorder="normal",
        ),
        yaxis=dict(
            showgrid=True, gridcolor="rgba(255,255,255,0.05)",
            color="#555d72",
        ),
        xaxis=dict(showgrid=False, color="#555d72"),
    )
    st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})
 
# ─────────────────────────────────────────────────────────────────────────────
# PIE DE PÁGINA
# ─────────────────────────────────────────────────────────────────────────────
 
st.markdown(
    f"""
    <div class="footer">
        v4 · 5 niveles de severidad · 3 horizontes (+1h, +3h, +6h) ·
        Modelos Random Forest independientes ·
        Datos reales Nuestra Señora de los Remedios — Cali (2024–2026) ·
        Actualizado al {datetime.now().strftime('%d/%m/%Y %H:%M')}
    </div>
    """,
    unsafe_allow_html=True,
)
 


