"""
retrain_model.py
================
Re-entrena 3 modelos Random Forest independientes (+1h, +3h, +6h)
con 5 niveles de severidad operativa y encoding ciclico.

Exporta:
  modelo_rf_1h.pkl, modelo_rf_3h.pkl, modelo_rf_6h.pkl
  config_v3.pkl       (umbrales, niveles, colores)
  features_v3.pkl     (lista de features)
  metricas_v3.csv     (metricas de validacion cruzada)
"""

import pandas as pd
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "dataset_train.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

UMBRALES = [1.0, 1.8, 2.8, 3.8]
NIVELES = ["Verde_Amarillo", "Presion_Moderada", "Presion_Alta", "Critico", "Colapso"]
COLORES = {"Verde_Amarillo": "#28a745", "Presion_Moderada": "#ffc107",
           "Presion_Alta": "#fd7e14", "Critico": "#dc3545", "Colapso": "#6f42c1"}
HORIZONTES = {"1h": 1, "3h": 3, "6h": 6}

BASE_FEATURES = [
    "cant_triage", "cola_esperando_inicio_hora", "camas_ocupadas",
    "abandonos_triage", "capacidad_hora",
    "cant_triage_lag1h", "cant_triage_lag2h", "cant_triage_lag3h",
    "saturacion_lag1h", "saturacion_lag3h", "saturacion_lag6h",
    "abandonos_lag1h", "cola_lag1h", "camas_ocupadas_lag1h",
    "tiempo_turnero_min", "pct_cumpl_oportunidad_triage",
    "pct_cumpl_consulta_t2", "tiempo_consulta_min",
    "es_festivo",
]
CICLICAS = ["hora_sin", "hora_cos", "dia_sin", "dia_cos"]
COMPUESTAS = ["presion_sistema"]
FEATURES = BASE_FEATURES + CICLICAS + COMPUESTAS
TARGET = "saturacion_idx"


def clasificar_severidad(valor):
    if valor < UMBRALES[0]:
        return "Verde_Amarillo"
    elif valor < UMBRALES[1]:
        return "Presion_Moderada"
    elif valor < UMBRALES[2]:
        return "Presion_Alta"
    elif valor < UMBRALES[3]:
        return "Critico"
    else:
        return "Colapso"


def feature_engineering(df):
    df["hora_sin"] = np.sin(2 * np.pi * df["hora"] / 24)
    df["hora_cos"] = np.cos(2 * np.pi * df["hora"] / 24)
    df["dia_sin"] = np.sin(2 * np.pi * df["dia_semana_enc"] / 7)
    df["dia_cos"] = np.cos(2 * np.pi * df["dia_semana_enc"] / 7)
    df["presion_sistema"] = df["cola_esperando_inicio_hora"] + df["camas_ocupadas"]
    return df


def mase(y_true, y_pred):
    y_naive = np.roll(y_true, 1)
    y_naive[0] = y_true[0]
    mae_model = mean_absolute_error(y_true, y_pred)
    mae_naive = mean_absolute_error(y_true, y_naive)
    return mae_model / mae_naive if mae_naive > 0 else np.nan


print("=" * 60)
print("Cargando datos...")
df = pd.read_csv(DATA_PATH)
df["datetime"] = pd.to_datetime(df["datetime"])
df = df.sort_values("datetime").reset_index(drop=True)
print(f"  Filas: {len(df):,}")

df = feature_engineering(df)

print("\nDistribucion del target (saturacion_idx):")
print(df[TARGET].describe())

print(f"\nUmbrales de severidad (5 niveles):")
for i, nivel in enumerate(NIVELES):
    if i == 0:
        print(f"  {nivel:20s}: < {UMBRALES[0]}")
    elif i == len(NIVELES) - 1:
        print(f"  {nivel:20s}: >= {UMBRALES[-1]}")
    else:
        print(f"  {nivel:20s}: {UMBRALES[i-1]} - {UMBRALES[i]}")

df["severidad"] = df[TARGET].apply(clasificar_severidad)
print("\nDistribucion severidad:")
for nivel in NIVELES:
    cnt = (df["severidad"] == nivel).sum()
    print(f"  {nivel:20s}: {cnt:>6,} ({cnt/len(df)*100:.1f}%)")

for nombre, horas in HORIZONTES.items():
    df[f"target_{nombre}"] = df[TARGET].shift(-horas)

df_model = df.dropna(subset=[f"target_{h}" for h in HORIZONTES]).copy()
print(f"\nFilas con target disponible: {len(df_model):,}")

X = df_model[FEATURES]
y_dict = {h: df_model[f"target_{h}"] for h in HORIZONTES}

tscv = TimeSeriesSplit(n_splits=5, gap=0)

modelos = {}
metricas_cv = []

for nombre, horas in HORIZONTES.items():
    print(f"\n{'=' * 60}")
    print(f"Entrenando modelo +{nombre}")
    y = y_dict[nombre]

    fold_scores = []
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        rf = RandomForestRegressor(
            n_estimators=300, max_depth=12, min_samples_leaf=5,
            n_jobs=-1, random_state=42 + fold
        )
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)

        mae = mean_absolute_error(y_val, y_pred)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        r2 = r2_score(y_val, y_pred)
        mase_val = mase(y_val.values, y_pred)

        fold_scores.append({"fold": fold + 1, "MAE": mae, "RMSE": rmse, "R2": r2, "MASE": mase_val})
        print(f"  Fold {fold + 1}: MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}, MASE={mase_val:.4f}")

    rf_final = RandomForestRegressor(
        n_estimators=300, max_depth=12, min_samples_leaf=5,
        n_jobs=-1, random_state=42
    )
    rf_final.fit(X, y)
    modelos[nombre] = rf_final

    avg = {k: np.mean([s[k] for s in fold_scores]) for k in ["MAE", "RMSE", "R2", "MASE"]}
    metricas_cv.append({"horizonte": nombre, **avg})
    print(f"  >> PROMEDIO CV: MAE={avg['MAE']:.4f}, R2={avg['R2']:.4f}, MASE={avg['MASE']:.4f}")

df_metricas = pd.DataFrame(metricas_cv)
print(f"\n{'=' * 60}")
print("RESUMEN CV - 3 MODELOS")
print(df_metricas.to_string(index=False))

# Metricas de clasificacion en test (ultimo fold)
from sklearn.metrics import balanced_accuracy_score, f1_score

_, test_idx = list(tscv.split(X))[-1]
X_test = X.iloc[test_idx]

print(f"\n{'=' * 60}")
print("METRICAS DE CLASIFICACION (5 niveles) — test holdout")
for nombre in HORIZONTES:
    y_test = y_dict[nombre].iloc[test_idx]
    y_pred = modelos[nombre].predict(X_test)
    y_test_cls = [clasificar_severidad(v) for v in y_test]
    y_pred_cls = [clasificar_severidad(v) for v in y_pred]
    ba = balanced_accuracy_score(y_test_cls, y_pred_cls)
    f1 = f1_score(y_test_cls, y_pred_cls, average="macro")
    print(f"  +{nombre}: Balanced Accuracy={ba:.4f}, F1-macro={f1:.4f}")

# Exportar artefactos
print(f"\n{'=' * 60}")
print("Exportando artefactos...")
for nombre in modelos:
    path = os.path.join(MODEL_DIR, f"modelo_rf_{nombre}.pkl")
    with open(path, "wb") as f:
        pickle.dump(modelos[nombre], f)
    print(f"  modelo_rf_{nombre}.pkl")

config = {
    "umbrales": UMBRALES,
    "niveles": NIVELES,
    "horizontes": list(HORIZONTES.keys()),
    "features": FEATURES,
    "colores": COLORES,
    "fecha_entrenamiento": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
}
with open(os.path.join(MODEL_DIR, "config_v3.pkl"), "wb") as f:
    pickle.dump(config, f)
print(f"  config_v3.pkl")

with open(os.path.join(MODEL_DIR, "features_v3.pkl"), "wb") as f:
    pickle.dump(FEATURES, f)
print(f"  features_v3.pkl")

df_metricas.to_csv(os.path.join(MODEL_DIR, "metricas_v3.csv"), index=False)
print(f"  metricas_v3.csv")
print(f"\nTodos los artefactos exportados en: {MODEL_DIR}")
print("=" * 60)
