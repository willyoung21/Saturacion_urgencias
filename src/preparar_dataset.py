"""
preparar_dataset.py
====================
Convierte las 3 exportaciones CSV del modelo semántico de Power BI/Fabric
en un dataset horario listo para entrenar el modelo de saturación de urgencias.

Inputs esperados (carpeta data/):
  - triage_metricas_horarias.csv  (Query 1 — extraer_datos.py)
  - urgencias_timestamps.csv      (Query 2 — extraer_datos.py)
  - triage_timestamps.csv         (Query 3 — extraer_datos.py)

Output:
  - data/dataset_train.csv        (dataset final con lags y target)

Uso:
  python src/preparar_dataset.py

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PREVENCIÓN DE DATA LEAKAGE  (documento de decisiones)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. FEATURES basadas en estado pasado:
   - camas_ocupadas(T)     = stock de pacientes activos ANTES de la hora T
                             (solo egresos que ocurrieron <= T)
   - cola_esperando(T)     = pacientes admitidos ANTES de T sin resolución a T
   - Todos los lags se calculan con .shift(n) sobre la serie cronológica:
     lag1h = valor de T-1, lag3h = T-3, lag6h = T-6
   → Nunca se usa información del futuro para construir el feature.

2. TARGET: saturacion_idx(T) = camas_ocupadas(T) / capacidad(T)
   Solo usa estado en T (no promedio futuro).

3. SPLIT TEMPORAL (aplicar en el notebook):
   - CORRECTO : cortar cronológicamente (últimos N días = test)
   - INCORRECTO: shuffle antes de split (mezclaría futuro con pasado)
   - Los lags se calculan sobre el dataset COMPLETO y luego se hace el split,
     de modo que las primeras filas del test set heredan lags del training.
     Esto es correcto porque en producción también conocemos el estado previo.

4. NORMALIZACIÓN (si se aplica):
   - Calcular media/std SOLO en el training set y aplicar al test set.

5. VALIDACIÓN CRUZADA:
   - Usar TimeSeriesSplit (sklearn) no KFold estándar.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# Capacidad de camillas de observación urgencias (lookup histórico)
CAPACIDAD_HISTORICA = [
    ("2026-01-01", 20),   # reducción desde enero 2026
    ("2024-01-01", 24),   # capacidad original (inicio de datos)
]

# Festivos colombianos 2024-2026 (calendario oficial)
FESTIVOS_CO = pd.to_datetime([
    # 2024
    "2024-01-01", "2024-01-08", "2024-03-25", "2024-03-28", "2024-03-29",
    "2024-04-01", "2024-05-01", "2024-05-13", "2024-06-03", "2024-06-10",
    "2024-07-01", "2024-07-04", "2024-07-20", "2024-08-07", "2024-08-19",
    "2024-10-14", "2024-11-04", "2024-11-11", "2024-12-08", "2024-12-25",
    # 2025
    "2025-01-01", "2025-01-06", "2025-03-24", "2025-04-17", "2025-04-18",
    "2025-05-01", "2025-06-02", "2025-06-23", "2025-06-30", "2025-07-20",
    "2025-08-07", "2025-08-18", "2025-10-13", "2025-11-03", "2025-11-17",
    "2025-12-08", "2025-12-25",
    # 2026
    "2026-01-01", "2026-01-12", "2026-03-23", "2026-04-02", "2026-04-03",
    "2026-05-01", "2026-05-18", "2026-06-08", "2026-06-15", "2026-06-29",
    "2026-07-20", "2026-08-07", "2026-08-17",
])
FESTIVOS_DATES = set(FESTIVOS_CO.date)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_flexible(series: pd.Series) -> pd.Series:
    """
    Parsea fechas/horas en formato mixto:
      - ISO 8601  : '2024-01-02T10:50:28'  (Power BI REST API)
      - Colombiano: '2/01/2024 10:50:28 a. m.'  (MCP DAX CSV)
      - Base 1899 : '1899-12-30T10:50:28'  (columnas de solo-hora en Power BI)
    Retorna una Serie de Timestamps. Los valores no parseables quedan NaT.
    """
    # Intento principal: pandas infiere el formato
    result = pd.to_datetime(series, infer_datetime_format=True, errors="coerce")
    # Fallback para formato colombiano (día primero)
    mask_nat = result.isna() & series.notna()
    if mask_nat.any():
        result[mask_nat] = pd.to_datetime(
            series[mask_nat], dayfirst=True, errors="coerce"
        )
    return result


def extract_time_component(hora_series: pd.Series) -> pd.Series:
    """
    Extrae el componente de tiempo de una columna 'hora' de Power BI.
    Power BI almacena solo-hora como datetime con base 1899-12-30.
    Devuelve Series de timedelta (duración desde medianoche).
    """
    dt = parse_flexible(hora_series)
    return (
        pd.to_timedelta(dt.dt.hour,   unit="h")
      + pd.to_timedelta(dt.dt.minute, unit="m")
      + pd.to_timedelta(dt.dt.second, unit="s")
    )


def combine_fecha_hora(fecha_col: pd.Series, hora_col: pd.Series) -> pd.Series:
    """
    Combina una columna de fecha (date part) y una columna de hora (Power BI
    base-1899 o timedelta) en un Timestamp completo.
    """
    fechas = parse_flexible(fecha_col).dt.normalize()
    delta  = extract_time_component(hora_col)
    return fechas + delta


def to_numeric_robust(series: pd.Series) -> pd.Series:
    """
    Convierte a numérico manejando comas decimales del locale colombiano
    (exportación MCP) y valores ya numéricos (exportación REST API).
    """
    if series.dtype == object:
        return pd.to_numeric(
            series.astype(str).str.replace(",", ".", regex=False),
            errors="coerce"
        )
    return pd.to_numeric(series, errors="coerce")


def get_capacidad(dt: pd.Timestamp) -> int:
    """Capacidad de camillas de observación urgencias vigente para la fecha."""
    for fecha_str, cap in CAPACIDAD_HISTORICA:
        if dt >= pd.Timestamp(fecha_str):
            return cap
    return CAPACIDAD_HISTORICA[-1][1]


# ---------------------------------------------------------------------------
# 1. Cargar datos
# ---------------------------------------------------------------------------

def cargar_datos():
    print("=" * 60)
    print("PASO 1 — Cargando archivos de entrada")
    print("=" * 60)

    paths = {
        "triage_h": DATA_DIR / "triage_metricas_horarias.csv",
        "urg_ts":   DATA_DIR / "urgencias_timestamps.csv",
        "tri_ts":   DATA_DIR / "triage_timestamps.csv",
    }
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(
                f"No encontrado: {path}\n"
                "Ejecuta primero:  python src/extraer_datos.py"
            )

    def _clean_cols(df: pd.DataFrame) -> pd.DataFrame:
        """Elimina prefijos de tabla y corchetes de los nombres de columna.
        'Triage[FECHA]' → 'FECHA',  '[cant_triage]' → 'cant_triage'."""
        df.columns = [
            c.split("[")[-1].rstrip("]").strip() for c in df.columns
        ]
        return df

    df_triage_h = _clean_cols(pd.read_csv(paths["triage_h"]))
    df_urg_ts   = _clean_cols(pd.read_csv(paths["urg_ts"]))
    df_tri_ts   = _clean_cols(pd.read_csv(paths["tri_ts"]))

    print(f"  triage_metricas_horarias : {len(df_triage_h):,} filas")
    print(f"  urgencias_timestamps     : {len(df_urg_ts):,} filas")
    print(f"  triage_timestamps        : {len(df_tri_ts):,} filas")
    return df_triage_h, df_urg_ts, df_tri_ts


# ---------------------------------------------------------------------------
# 2. Construir índice horario base
# ---------------------------------------------------------------------------

def construir_serie_horaria(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parsea FECHA + Hora → datetime completo y normaliza columnas numéricas.
    Maneja tanto el formato de exportación MCP (comas decimales, nombres con
    prefijo de tabla) como el formato REST API (ISO dates, nombres limpios).
    """
    df = df.copy()
    df.columns = df.columns.str.strip()

    # Detectar columna de fecha y hora (tolerante a variantes de nombre)
    fecha_col = next(
        c for c in df.columns
        if c.upper() in ("FECHA", "TRIAGE[FECHA]") or "FECHA" in c.upper()
    )
    hora_col = next(
        c for c in df.columns
        if c.lower() in ("hora", "triage[hora]", "[hora]")
    )

    # Construir datetime: "YYYY-MM-DD HH:00:00"
    # Extraemos solo la parte de fecha (primeros 10 caracteres sirven para
    # 'DD/MM/YYYY' o 'YYYY-MM-DD') y la hora como entero 0-23
    fechas_str = parse_flexible(df[fecha_col]).dt.strftime("%Y-%m-%d")
    horas_int  = pd.to_numeric(df[hora_col], errors="coerce").fillna(0).astype(int)
    df["datetime"] = pd.to_datetime(
        fechas_str + " " + horas_int.astype(str).str.zfill(2) + ":00:00"
    )

    # Normalizar columnas numéricas (manejo locale colombiano vs ISO)
    for col in [
        "cant_triage", "abandonos_triage",
        "tiempo_turnero_min", "tiempo_consulta_min",
        "pct_cumpl_oportunidad_triage", "pct_cumpl_consulta_t2",
    ]:
        if col in df.columns:
            df[col] = to_numeric_robust(df[col])

    # cant_triage y abandonos_triage como enteros (con NaN → 0 para abandonos)
    df["cant_triage"]      = df["cant_triage"].fillna(0).astype(int)
    df["abandonos_triage"] = df["abandonos_triage"].fillna(0).astype(int)

    return df.sort_values("datetime").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. camas_ocupadas por hora  [SIN DATA LEAKAGE]
# ---------------------------------------------------------------------------

def calcular_camas_ocupadas(df_urg: pd.DataFrame, idx_horario: pd.DatetimeIndex) -> pd.Series:
    """
    Para cada timestamp T del índice horario, cuenta cuántos pacientes tenían
    cama activa: ingreso_dt <= T  AND  (egreso_dt IS NaT OR egreso_dt > T).

    LEAKAGE-FREE: usa solo eventos PASADOS respecto a T.
    Algoritmo: eventos +1/-1 + cumsum + merge_asof → O(n log n).
    """
    print("  Calculando camas_ocupadas por hora...")
    df_urg = df_urg.copy()
    df_urg.columns = df_urg.columns.str.strip()

    df_urg["ingreso_dt"] = combine_fecha_hora(df_urg["fecha_ingreso"], df_urg["hora_ingreso"])
    df_urg["egreso_dt"]  = combine_fecha_hora(df_urg["fecha_egreso"],  df_urg["hora_egreso"])

    # Filas con ingreso inválido se ignoran
    df_urg = df_urg.dropna(subset=["ingreso_dt"])

    ingresos = pd.DataFrame({"dt": df_urg["ingreso_dt"], "delta": 1})
    egresos  = pd.DataFrame({"dt": df_urg["egreso_dt"].dropna(), "delta": -1})
    eventos  = (
        pd.concat([ingresos, egresos])
        .groupby("dt")["delta"].sum()
        .reset_index()
        .sort_values("dt")
    )
    eventos["cumsum"] = eventos["delta"].cumsum()

    df_hrs = pd.DataFrame({"datetime": idx_horario}).sort_values("datetime")
    merged = pd.merge_asof(df_hrs, eventos, left_on="datetime", right_on="dt")
    merged["cumsum"] = merged["cumsum"].fillna(0).clip(lower=0)
    return merged.set_index("datetime")["cumsum"].astype(int)


# ---------------------------------------------------------------------------
# 4. cola_esperando por hora  [SIN DATA LEAKAGE]
# ---------------------------------------------------------------------------

def calcular_cola_esperando(df_ts: pd.DataFrame, idx_horario: pd.DatetimeIndex) -> pd.Series:
    """
    Para cada timestamp T, cuenta pacientes admitidos ANTES de T cuya consulta
    aún no comenzó (resolucion_dt > T o es NaT).

    LEAKAGE-FREE: usa solo admisiones ocurridas antes de T.
    """
    print("  Calculando cola_esperando_inicio_hora...")
    df_ts = df_ts.copy()
    df_ts.columns = df_ts.columns.str.strip()

    df_ts["admision_dt"]   = combine_fecha_hora(df_ts["fecha_admision"], df_ts["hora_admision"])
    df_ts["resolucion_dt"] = parse_flexible(df_ts["fecha_resolucion"])

    df_ts = df_ts.dropna(subset=["admision_dt"])

    admisiones   = pd.DataFrame({"dt": df_ts["admision_dt"], "delta": 1})
    resoluciones = pd.DataFrame({"dt": df_ts["resolucion_dt"].dropna(), "delta": -1})
    eventos = (
        pd.concat([admisiones, resoluciones])
        .groupby("dt")["delta"].sum()
        .reset_index()
        .sort_values("dt")
    )
    eventos["cumsum"] = eventos["delta"].cumsum()

    df_hrs = pd.DataFrame({"datetime": idx_horario}).sort_values("datetime")
    merged = pd.merge_asof(df_hrs, eventos, left_on="datetime", right_on="dt")
    merged["cumsum"] = merged["cumsum"].fillna(0).clip(lower=0)
    return merged.set_index("datetime")["cumsum"].astype(int)


# ---------------------------------------------------------------------------
# 5. Pipeline principal
# ---------------------------------------------------------------------------

FEATURES = [
    # Estado actual del sistema
    "cant_triage", "cola_esperando_inicio_hora", "camas_ocupadas",
    "abandonos_triage", "capacidad_hora",
    # Lags — estado pasado (PREVIENE LEAKAGE: shift hacia atrás)
    "cant_triage_lag1h", "cant_triage_lag2h", "cant_triage_lag3h",
    "saturacion_lag1h",  "saturacion_lag3h",  "saturacion_lag6h",
    "abandonos_lag1h",   "cola_lag1h",         "camas_ocupadas_lag1h",
    # Calidad de atención
    "tiempo_turnero_min", "pct_cumpl_oportunidad_triage",
    "pct_cumpl_consulta_t2", "tiempo_consulta_min",
    # Estacionalidad (no contiene info futura)
    "hora", "dia_semana_enc", "es_festivo",
]
TARGET = "saturacion_idx"


def main():
    # ── Carga ──────────────────────────────────────────────────────────────────
    df_triage_h, df_urg_ts, df_tri_ts = cargar_datos()

    print("\nPASO 2 — Construyendo serie horaria base")
    df = construir_serie_horaria(df_triage_h)
    idx = pd.DatetimeIndex(df["datetime"])
    print(f"  Serie horaria: {df['datetime'].min()} → {df['datetime'].max()} ({len(df):,} filas)")

    # ── Métricas operativas calculadas [leakage-free] ─────────────────────────
    print("\nPASO 3 — Calculando métricas operativas")
    camas = calcular_camas_ocupadas(df_urg_ts, idx)
    cola  = calcular_cola_esperando(df_tri_ts, idx)

    df = df.set_index("datetime")
    df["camas_ocupadas"]             = camas.reindex(df.index).fillna(0).astype(int)
    df["cola_esperando_inicio_hora"] = cola.reindex(df.index).fillna(0).astype(int)
    df = df.reset_index()

    # ── Capacidad + target ────────────────────────────────────────────────────
    print("\nPASO 4 — Calculando target y features de tiempo")
    df["capacidad_hora"]  = df["datetime"].apply(get_capacidad)
    df["saturacion_idx"]  = (df["camas_ocupadas"] / df["capacidad_hora"]).round(4)
    df["nivel_saturacion"] = pd.cut(
        df["saturacion_idx"],
        bins=[-np.inf, 0.7, 1.0, np.inf],
        labels=["Verde", "Amarillo", "Rojo"]
    ).astype(str)

    # ── Estacionalidad ────────────────────────────────────────────────────────
    orden_carga = {
        "Monday": 6, "Tuesday": 5, "Wednesday": 4, "Thursday": 3,
        "Friday": 2, "Saturday": 1, "Sunday": 0,
    }
    df["hora"]           = df["datetime"].dt.hour
    df["dia_semana"]     = df["datetime"].dt.day_name()
    df["dia_semana_enc"] = df["dia_semana"].map(orden_carga)
    df["es_festivo"]     = df["datetime"].dt.date.isin(FESTIVOS_DATES).astype(int)

    # ── Lags [LEAKAGE-FREE: calculados en serie completa cronológica] ─────────
    # IMPORTANTE: los lags se calculan ANTES del split train/test.
    # Esto es correcto: en producción, al predecir hora T, sí conocemos T-1, T-3, T-6.
    # Las primeras 6 filas resultarán NaN (lag6h sin dato previo) y se eliminan con dropna().
    print("  Calculando lags temporales...")
    df = df.sort_values("datetime").reset_index(drop=True)

    lag_map = {
        "saturacion_lag1h":     ("saturacion_idx", 1),
        "saturacion_lag3h":     ("saturacion_idx", 3),
        "saturacion_lag6h":     ("saturacion_idx", 6),
        "cant_triage_lag1h":    ("cant_triage",    1),
        "cant_triage_lag2h":    ("cant_triage",    2),
        "cant_triage_lag3h":    ("cant_triage",    3),
        "abandonos_lag1h":      ("abandonos_triage", 1),
        "cola_lag1h":           ("cola_esperando_inicio_hora", 1),
        "camas_ocupadas_lag1h": ("camas_ocupadas", 1),
    }
    for new_col, (src_col, lag) in lag_map.items():
        df[new_col] = df[src_col].shift(lag)

    # ── Selección y exportación ───────────────────────────────────────────────
    print("\nPASO 5 — Exportando dataset_train.csv")

    # Imputar NaN en features que son NaN en horas sin pacientes de cierto tipo:
    # - tiempo_consulta_min / pct_cumpl_consulta_t2: no hay triage 2 en esa hora → 0
    # - tiempo_turnero_min / pct_cumpl_oportunidad_triage: ídem → media histórica
    impute_zero = ["tiempo_consulta_min", "pct_cumpl_consulta_t2", "abandonos_triage"]
    for col in impute_zero:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    impute_median = ["tiempo_turnero_min", "pct_cumpl_oportunidad_triage"]
    for col in impute_median:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    df_final = df[FEATURES + [TARGET, "datetime", "nivel_saturacion"]].dropna()
    df_final = df_final.reset_index(drop=True)

    out_path = DATA_DIR / "dataset_train.csv"
    df_final.to_csv(out_path, index=False, encoding="utf-8")

    # ── Reporte ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("DATASET LISTO")
    print("=" * 60)
    print(f"  Archivo    : {out_path}")
    print(f"  Filas      : {len(df_final):,}")
    print(f"  Features   : {len(FEATURES)}")
    print(f"  Período    : {df_final['datetime'].min().date()} → {df_final['datetime'].max().date()}")
    print(f"  Target     : {df_final[TARGET].min():.3f} – {df_final[TARGET].max():.3f}  "
          f"(media={df_final[TARGET].mean():.3f})")

    print("\nDistribución semáforo:")
    dist = df_final["nivel_saturacion"].value_counts()
    for lvl, cnt in dist.items():
        print(f"  {lvl:<10}: {cnt:>6,}  ({cnt/len(df_final)*100:.1f}%)")

    nulos = df_final[FEATURES + [TARGET]].isnull().sum()
    if nulos.sum() == 0:
        print("\nNulos: ninguno ✔")
    else:
        print("\nNulos por columna (revisar):")
        print(nulos[nulos > 0].to_string())

    print("\nSugerido para split en el notebook:")
    cutoff = df_final["datetime"].max() - pd.Timedelta(days=90)
    n_train = (df_final["datetime"] <= cutoff).sum()
    n_test  = (df_final["datetime"] >  cutoff).sum()
    print(f"  Train : hasta {cutoff.date()}  ({n_train:,} filas)")
    print(f"  Test  : desde {cutoff.date()}  ({n_test:,} filas)")
    print("=" * 60)


if __name__ == "__main__":
    main()
