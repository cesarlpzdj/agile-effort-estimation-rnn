# Effort Estimator: Motor de Aprendizaje Profundo y API de Inferencia

Este módulo constituye el núcleo de procesamiento de lenguaje natural (PLN), entrenamiento de redes neuronales recurrentes (LSTM) y servicio de inferencia en tiempo real para la estimación de esfuerzo ágil en Story Points.

Forma parte del trabajo de investigación:
> **Empirical Evaluation of Recurrent Neural Networks for Agile Effort Estimation**  
> *Revista Objetivo:* Empirical Software Engineering (Springer Nature)  
> *Autor:* César López de Jesús  
> *Afiliación:* Maestría en Inteligencia Artificial y Analítica de Datos (UACJ)

---

## 1. Arquitectura del Modelo (`EMB_LSTM_v2`)

El modelo neuronal implementado en este módulo supera los paradigmas tradicionales basados en frecuencias de términos estáticas (TF-IDF) mediante una representación secuencial continua:

* **Preprocesamiento:** Concatenación de título y descripción del requerimiento, normalización tipográfica, eliminación de *stopwords* en inglés y lematización morfológica con WordNet.
* **Vectorización Secuencial (`TextVectorization`):** Vocabulario adaptado de $V = 20,000$ tokens más frecuentes, con longitud de secuencia fija de $L = 100$ palabras (truncamiento y relleno con padding a cero).
* **Capa de Embedding Semántico:** Matriz entrenable de $20,000 \times 128$ dimensiones con `mask_zero=True` para propagar el enmascaramiento temporal a lo largo de las secuencias.
* **Capa Recurrente:** `LSTM` unidirigida con 64 unidades ocultas y función de activación `tanh`.
* **Regularización:** Capa de `Dropout(rate=0.3)` aplicada sobre las salidas de la celda LSTM.
* **Capas Densas de Proyección:** Capa intermedia `Dense(32, activation='relu')` seguida de la neurona de salida lineal `Dense(1, activation='linear')`.
* **Objetivo Matemático:** Transformación en escala logarítmica $\log(1 + y)$ con función de pérdida **Huber ($\delta = 1.0$)**, la cual previene la inestabilidad por gradientes explosivos ante historias de usuario atípicas o extremas ($\ge 13$ SP) sin perder sensibilidad lineal en errores moderados. La predicción se invierte a la escala original mediante $\hat{y} = \max(0, \exp(\hat{z}) - 1)$.

---

## 2. Estructura de Archivos del Módulo

```
effort-estimator/
├── api.py                           # Microservicio FastAPI con endpoint /api/v1/predict
├── fase2_lstm_v2.py                 # Pipeline de entrenamiento completo y evaluación global
├── diagnostico_epoca2.py            # Análisis de convergencia y prueba de Kolmogorov-Smirnov
├── experimentos_etapa2.py           # Runner de la matriz factorial 2x2x2 en 5 semillas
├── analisis_etapa3_proyectos.py     # Pruebas de Wilcoxon por proyecto y correlación de Spearman
├── crear_paquete_replicacion.py     # Script generador de los splits maestros de replicación
├── requirements.txt                 # Dependencias fijadas de Python
├── pruebas.http                     # Peticiones HTTP de prueba para depuración REST
├── dataset/                         # 16 archivos CSV con tickets de Jira del benchmark Deep-SE
│   ├── moodle.csv
│   ├── mesos.csv
│   ├── springxd.csv
│   └── ... (16 proyectos)
└── artefactos_exportados/           # Modelos compilados y resultados de evaluación
    ├── v2/temporal/                 # Modelo entrenado v2 (Keras), vocabulario JSON y métricas
    ├── etapa2/                      # Resultados de la matriz factorial y diagnóstico JSON
    └── etapa3/                      # Tablas estadísticas de RQ3 por proyecto
```

---

## 3. Instalación y Configuración

### Requisitos Previos
* Python 3.10, 3.11 o 3.12.
* Herramienta de entornos virtuales (`venv` o `uv`).

### Paso a Paso
```bash
# 1. Posicionarse en el directorio del estimador
cd effort-estimator

# 2. Crear y activar el entorno virtual
python3 -m venv venv
source venv/bin/activate

# 3. Instalar las dependencias fijadas
pip install -r requirements.txt

# 4. Descargar corpus léxicos necesarios de NLTK
python3 -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"
```

---

## 4. Servicio de Inferencia en Tiempo Real (`api.py`)

El archivo `api.py` expone un microservicio REST de alta velocidad construido con **FastAPI** y **Uvicorn**, diseñado para integrarse con tableros de gestión ágil (Jira, GitHub Issues) o con el cliente React de soporte a la decisión (`frontend-estimator`).

### Ejecución del Servidor
```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```
* **URL Base:** `http://localhost:8000`
* **Documentación Interactiva (Swagger/OpenAPI):** `http://localhost:8000/docs`
* **Especificación ReDoc:** `http://localhost:8000/redoc`

### Contrato de la API (`POST /api/v1/predict`)

#### Ejemplo de Petición (`Request Body`)
```json
{
  "ticket_id": "MDL-65230",
  "project_domain": "moodle",
  "title": "Fix quiz submission timeout under heavy concurrent loads",
  "description": "Students experience network socket timeouts when submitting quiz answers simultaneously during final exam periods. Large file attachments exacerbate the database lock contention.",
  "issue_type": "Bug"
}
```

#### Ejemplo de Respuesta (`Response Body`)
```json
{
  "ticket_id": "MDL-65230",
  "prediction": {
    "story_points_raw": 3.42,
    "story_points_rounded": 3,
    "historical_mae": 3.26,
    "expected_mae": 3.26,
    "confidence_interval": "± 3.26 SP (MAE global histórico)",
    "coverage_note": "El MAE global histórico cubre el 72.6% de los tickets en prueba temporal; no representa un intervalo de incertidumbre individual",
    "rmse_temporal": 7.01,
    "project_median_baseline": 8.0,
    "model_version": "emb_lstm_v2_temporal"
  },
  "status": "success"
}
```

* **Latencia:** $\sim 23.3$ ms en promedio (p95: $25.1$ ms), cumpliendo con creces los requerimientos de interacción interactiva sub-100 ms.
* **Soporte Dual:** Retorna en paralelo la estimación del modelo neuronal (`story_points_rounded`) y la mediana histórica del proyecto (`project_median_baseline`: Clover 2.0, Jira 3.0, Moodle 8.0, etc.), permitiendo que la interfaz alerte discrepancias mayores a 2 SP para revisión técnica durante el Planning Poker.

---

## 5. Ejecución de los Scripts Experimentales

Todos los scripts cuentan con aislamiento de rutas y pueden ejecutarse directamente:

### 1. Diagnóstico de la "Época 2" y Shift Temporal (Subetapa 2a)
Evalúa las curvas de aprendizaje y ejecuta la prueba Kolmogorov-Smirnov entre los splits cronológicos de los proyectos:
```bash
python diagnostico_epoca2.py
```
*Salida:* `artefactos_exportados/etapa2/diagnostico_epoca2.json`.

### 2. Matriz Factorial $2 \times 2 \times 2$ en 5 Semillas y Arquitectura Bimodal (Subetapa 2b)
Ejecuta 50 entrenamientos independientes cruzando preprocesamiento, escala y función de pérdida, más el modelo bimodal condicionado por embeddings de proyecto:
```bash
python experimentos_etapa2.py
```
*Salidas:* `artefactos_exportados/etapa2/runs_factoriales.csv` y `resumen_factoriales.csv`.

### 3. Análisis por Proyecto y Correlaciones de Spearman (Etapa 3 - RQ3)
Ejecuta las pruebas de Wilcoxon pareadas con ajuste de Holm-Bonferroni para los 16 proyectos y calcula las correlaciones no paramétricas de Spearman frente a las características del repositorio:
```bash
python analisis_etapa3_proyectos.py
```
*Salidas:* `artefactos_exportados/etapa3/tabla_proyectos_detallada.csv` y `correlaciones_spearman.csv`.

