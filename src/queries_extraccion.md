# Queries DAX — Extracción Dataset Real de Urgencias

**Modelo semántico:** `Ospedale Mensual [Dataset]`  
**Workspace Fabric:** `[3] CMI [PROD]`  
**Clínica:** Nuestra Cali (`805023423CL`)  
**Rango disponible:** 2024-01-01 → 2026-05-29  

Ejecuta cada query desde Fabric → Dataset → "Ejecutar consultas DAX" y descarga el resultado como CSV con el nombre indicado.

---

## Query 1 — Métricas Triage Horarias

**Archivo destino:** `data/triage_metricas_horarias.csv`  
Devuelve una fila por fecha + hora con todos los indicadores de triage.

```dax
EVALUATE
SUMMARIZECOLUMNS(
    'Triage'[FECHA],
    'Triage'[Hora],
    FILTER(ALL('Triage'), 'Triage'[Nit_Clinica] = "805023423CL"),
    FILTER(ALL('Triage'), 'Triage'[FECHA] >= DATE(2024, 1, 1)),

    -- Volumen de atención (excluye Triage 0 = sin clasificar, Triage 1 = reanimación)
    "cant_triage",
        CALCULATE(
            COUNTROWS('Triage'),
            'Triage'[Cla_Triage] <> 0,
            'Triage'[Cla_Triage] <> 1
        ),

    -- Abandonos post-triage en esa hora
    "abandonos_triage",
        CALCULATE(
            COUNTROWS('Triage'),
            'Triage'[Tipo_Abandono_Triage] = "Abandono Post-Triage"
        ),

    -- Tiempo promedio en turnero digital (digiturno) — minutos
    "tiempo_turnero_min",
        CALCULATE(
            AVERAGE('Triage'[Oportunidad_Digiturno]),
            'Triage'[Cla_Triage] <> 1
        ),

    -- Tiempo promedio hasta consulta médica — minutos
    "tiempo_consulta_min",
        CALCULATE(
            AVERAGE('Triage'[Oportunidad_Consulta]),
            'Triage'[Marca] = "Urgencia"
        ),

    -- % cumplimiento oportunidad triage (≤15 min, excluye T0 y T1)
    "pct_cumpl_oportunidad_triage",
        DIVIDE(
            CALCULATE(
                SUM('Triage'[Cumplimiento Oportunidad Triage]),
                'Triage'[Cla_Triage] <> 0,
                'Triage'[Cla_Triage] <> 1
            ),
            CALCULATE(
                COUNTROWS('Triage'),
                'Triage'[Cla_Triage] <> 0,
                'Triage'[Cla_Triage] <> 1
            )
        ),

    -- % cumplimiento tiempo consulta para Triage 2 (emergencia)
    "pct_cumpl_consulta_t2",
        DIVIDE(
            CALCULATE(
                SUM('Triage'[Cumplimiento Consulta Urgencias (Triage 2 y 3)]),
                'Triage'[Cla_Triage] = 2,
                'Triage'[Marca] = "Urgencia"
            ),
            CALCULATE(
                COUNTROWS('Triage'),
                'Triage'[Cla_Triage] = 2,
                'Triage'[Marca] = "Urgencia"
            )
        )
)
ORDER BY [FECHA], [Hora]
```

---

## Query 2 — Timestamps de Observación (Urgencias)

**Archivo destino:** `data/urgencias_timestamps.csv`  
Una fila por episodio de observación. Se usa para calcular `camas_ocupadas` por hora en Python.

```dax
EVALUATE
SELECTCOLUMNS(
    FILTER(
        '*Urgencias',
        '*Urgencias'[Nit_Clinica] = "805023423CL"
        && '*Urgencias'[Fecha Ingreso Pabellon] >= DATE(2024, 1, 1)
    ),
    "cnsc_ing",         '*Urgencias'[Cnsc_Ing],
    "fecha_ingreso",    '*Urgencias'[Fecha Ingreso Pabellon],
    "hora_ingreso",     '*Urgencias'[Hora Ingreso Pab],
    "fecha_egreso",     '*Urgencias'[Fecha Egreso Pabellon],
    "hora_egreso",      '*Urgencias'[Hora Egreso Pab],
    "tipo_estancia",    '*Urgencias'[Tipo Estancia],
    "horas_estancia",   '*Urgencias'[Sum Horas Estancia]
)
ORDER BY [fecha_ingreso], [hora_ingreso]
```

---

## Query 3 — Timestamps de Admisión Triage (para cola)

**Archivo destino:** `data/triage_timestamps.csv`  
Una fila por paciente. Se usa en Python para calcular `cola_esperando_inicio_hora` (stock de pacientes en espera al inicio de cada hora).

```dax
EVALUATE
SELECTCOLUMNS(
    FILTER(
        'Triage',
        'Triage'[Nit_Clinica] = "805023423CL"
        && 'Triage'[FECHA] >= DATE(2024, 1, 1)
        && 'Triage'[Cla_Triage] <> 0
        && 'Triage'[Cla_Triage] <> 1
    ),
    "cnsc_ing",              'Triage'[Cnsc_Ing],
    "fecha_triage",          'Triage'[FECHA],
    "cla_triage",            'Triage'[Cla_Triage],
    "fecha_admision",        'Triage'[Admision - Fecha],
    "hora_admision",         'Triage'[Admision - Hora],
    "fecha_resolucion",      'Triage'[Fecha_ResolucionC],
    "fecha_ingreso_consulta",'Triage'[Fecha_Ingreso_Consulta]
)
ORDER BY [fecha_admision], [hora_admision]
```

---

## Notas

| Campo | Fuente | Cálculo |
|-------|--------|---------|
| `cant_triage` | Query 1 | Directo |
| `abandonos_triage` | Query 1 | Directo |
| `tiempo_turnero_min` | Query 1 | Directo |
| `tiempo_consulta_min` | Query 1 | Directo |
| `pct_cumpl_oportunidad_triage` | Query 1 | Directo |
| `pct_cumpl_consulta_t2` | Query 1 | Directo |
| `camas_ocupadas` | Query 2 | Python — stock de pacientes activos por hora |
| `cola_esperando_inicio_hora` | Query 3 | Python — pacientes admitidos sin resolución por hora |
| `capacidad_hora` | Hardcoded | 24 camillas hasta 2025-12-31, 20 desde 2026-01-01 |
| `saturacion_idx` | Derivado | `camas_ocupadas / capacidad_hora` |
| `hora`, `dia_semana_enc`, `es_festivo` | Derivado | Python desde `datetime` |
| Lags (lag1h, lag3h, lag6h) | Derivado | Python — `.shift()` sobre la serie ordenada |
