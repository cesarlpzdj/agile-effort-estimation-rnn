import os
import re
import json
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
import tensorflow as tf
import keras
from keras.layers import TextVectorization
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
from fastapi.middleware.cors import CORSMiddleware

# 1. Configuración inicial y silenciamiento de logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

# 2. Definición de los Esquemas de Datos (Contratos JSON)
class TicketRequest(BaseModel):
    ticket_id: str
    project_domain: Optional[str] = "unknown"
    title: str
    description: str
    issue_type: Optional[str] = "Task"

class PredictionResponse(BaseModel):
    ticket_id: str
    prediction: Dict[str, Any]
    status: str

# 3. Inicialización de la Aplicación FastAPI
app = FastAPI(
    title="API Estimador de Esfuerzo Ágil (Deep Learning v2)",
    description="Microservicio predictivo basado en Embedding + LSTM secuencial y objetivo log1p para Agile Story Points",
    version="2.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # En producción se restringe al origen del frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mediana histórica por proyecto del benchmark Deep-SE (soporte a la decisión empírico)
# Derivadas de los datos previos a la prueba (80% histórico: train + val, idéntico a la evaluación del artículo)
PROJECT_MEDIANS = {
    "appcelerator": 5.0,
    "appceleratorstudio": 5.0,
    "aptanastudio": 8.0,
    "bamboo": 2.0,
    "clover": 2.0,
    "datamanagement": 4.0,
    "duracloud": 1.0,
    "jirasoftware": 3.0,
    "mesos": 3.0,
    "moodle": 8.0,
    "mule": 5.0,
    "mulestudio": 5.0,
    "springxd": 3.0,
    "talenddataquality": 5.0,
    "talendesb": 2.0,
    "titanium": 5.0,
    "usergrid": 3.0,
}

GLOBAL_MEDIAN = 4.0

# 4. Carga Global de Artefactos v2 (Modelo secuencial + TextVectorization)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "artefactos_exportados", "v2", "temporal")
print(f"Cargando artefactos v2 desde {MODEL_DIR}...")

try:
    with open(os.path.join(MODEL_DIR, "vocabulario_v2.json"), "r", encoding="utf-8") as f:
        vocab = json.load(f)

    with open(os.path.join(MODEL_DIR, "config_v2.json"), "r", encoding="utf-8") as f:
        model_config = json.load(f)

    max_len = model_config.get("max_len", 100)
    vocab_size = model_config.get("vocab_size", 20000)

    # Reconstrucción de la capa TextVectorization idéntica a la fase de entrenamiento
    vectorizer = TextVectorization(
        max_tokens=vocab_size,
        output_mode="int",
        output_sequence_length=max_len,
        standardize=None,
        split="whitespace"
    )
    vectorizer.set_vocabulary(vocab)

    modelo_lstm = keras.models.load_model(os.path.join(MODEL_DIR, "modelo_emb_lstm_v2.keras"))
    print("¡Modelos v2 cargados exitosamente!")
except Exception as e:
    print(f"Error crítico al cargar los artefactos v2: {e}")
    vectorizer = None
    modelo_lstm = None

# 5. Funciones Auxiliares
_STOP_WORDS = set(stopwords.words("english"))
_LEMMATIZER = WordNetLemmatizer()

def preprocess_text(text: str) -> str:
    """Limpieza de texto estandarizada (minúsculas, filtrado alfanumérico, stopwords y lematización)"""
    text = re.sub(r"[^a-zA-Z\s]", "", str(text).lower())
    tokens = [t for t in text.split() if t not in _STOP_WORDS]
    return " ".join(_LEMMATIZER.lemmatize(t) for t in tokens)

def snap_to_fibonacci(value: float) -> int:
    """Ajusta la predicción continua al valor más cercano de la secuencia ágil"""
    fibonacci_sequence = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
    return min(fibonacci_sequence, key=lambda x: abs(x - value))

# 6. Endpoint Principal
@app.post("/api/v1/predict", response_model=PredictionResponse)
async def predict_effort(ticket: TicketRequest):
    if modelo_lstm is None or vectorizer is None:
        raise HTTPException(status_code=503, detail="El modelo v2 no está disponible en memoria.")

    try:
        # A. Concatenación y limpieza de texto
        texto_completo = f"{ticket.title} {ticket.description}"
        texto_limpio = preprocess_text(texto_completo)

        # B. Tokenización secuencial mediante TextVectorization adaptado
        token_indices = vectorizer([texto_limpio]).numpy()

        # C. Inferencia neuronal sobre objetivo log1p
        pred_log1p = float(modelo_lstm.predict(token_indices, verbose=0)[0][0])

        # D. Transformación inversa (expm1 con cota inferior en 0)
        prediccion_raw = float(np.clip(np.expm1(pred_log1p), 0.0, None))

        # E. Baseline ingenuo por proyecto para contraste empírico
        proj_key = (ticket.project_domain or "").strip().lower()
        project_baseline = PROJECT_MEDIANS.get(proj_key, GLOBAL_MEDIAN)

        # F. Construcción de la respuesta
        respuesta = {
            "ticket_id": ticket.ticket_id,
            "prediction": {
                "story_points_raw": round(prediccion_raw, 2),
                "story_points_rounded": snap_to_fibonacci(prediccion_raw),
                "historical_mae": 3.26,
                "expected_mae": 3.26,
                "confidence_interval": "± 3.26 SP (MAE global histórico)",
                "coverage_note": "El MAE global histórico cubre el 72.6% de los tickets en prueba temporal; no representa un intervalo de incertidumbre individual",
                "rmse_temporal": 7.01,
                "project_median_baseline": project_baseline,
                "model_version": "emb_lstm_v2_temporal"
            },
            "status": "success"
        }

        return respuesta

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))