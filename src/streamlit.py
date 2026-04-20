import joblib
import os

# Forma segura de cargar la ruta
model_path = os.path.join('models', 'modelo_ia.pkl')
model = joblib.load(model_path)