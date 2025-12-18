import os
import json
import time
import pandas as pd
from flask import Flask, request, jsonify
from prometheus_client import start_http_server, Counter, Summary
import mlflow.pyfunc

# Konfigurasi MLOps dan Monitoring
DAGSHUB_MODEL_URI = "runs:/3aaa60c6501c4abfbc82ddf531f13000/model" 

# Set kredensial DagsHub
os.environ['MLFLOW_TRACKING_URI'] = "https://dagshub.com/Sabrinayusrina03/eksperimen_SML_SabrinaYusrina.mlflow"
os.environ['MLFLOW_TRACKING_USERNAME'] = "Sabrinayusrina03"
os.environ['MLFLOW_TRACKING_PASSWORD'] = os.environ.get("DAGSHUB_TOKEN")

#  Inisialisasi Metrik Prometheus (Minimal 10 Metrik)

# Counter: Menghitung total permintaan
REQUEST_COUNT = Counter(
    'http_requests_total', 
    'Total Request Count', 
    ['method', 'endpoint']
)
# Summary: Menghitung latensi/waktu pemrosesan
REQUEST_LATENCY = Summary(
    'http_request_latency_seconds', 
    'Request latency in seconds', 
    ['method', 'endpoint']
)
# Metrik Khusus ML
PREDICTION_COUNT = Counter('prediction_total', 'Total number of successful predictions')
PREDICTION_FAILURE_COUNT = Counter('prediction_failure_total', 'Total number of failed predictions')
INPUT_DATA_MISSING = Counter('input_data_missing_total', 'Total missing/null values in input')
INVALID_INPUT_TYPE = Counter('invalid_input_type_total', 'Total invalid data type in input')
PREDICTION_TIME = Summary('prediction_time_seconds', 'Model prediction time in seconds')
# Metrik Kebutuhan Alerting/Monitoring Lanjutan
GPU_REQUESTED_COUNT = Counter('gpu_requested_total', 'Count of requests where GPU is present')
SCREEN_SIZE_AVG = Summary('screen_size_avg', 'Average screen size in request')
RAM_USED_AVG = Summary('ram_used_avg', 'Average RAM in request')

# Setup Aplikasi dan Model

app = Flask(__name__)

try:
    start_http_server(8000, addr='0.0.0.0')
    print("Metrics server is running on port 8000")
except Exception as e:
    print(f"Metrics server error: {e}")

# Muat model dari DagsHub
print(f"Loading model from DagsHub: {DAGSHUB_MODEL_URI}...")
try:
    model = mlflow.pyfunc.load_model(DAGSHUB_MODEL_URI)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Endpoint Inferensi (POST)

@app.route('/predict', methods=['POST'])
@REQUEST_LATENCY.labels(method='POST', endpoint='/predict').time()
def predict():
    REQUEST_COUNT.labels(method='POST', endpoint='/predict').inc() 
    
    if model is None:
        PREDICTION_FAILURE_COUNT.inc()
        return jsonify({"error": "Model failed to load, cannot predict."}), 500

    try:
        data = request.json
        if not data:
            PREDICTION_FAILURE_COUNT.inc()
            return jsonify({"error": "No JSON data provided"}), 400
        
        # Konversi input JSON ke DataFrame untuk prediksi
        df = pd.DataFrame(data, index=[0]) 

        # Pengecekan Kualitas Data (Data Validation)
        if 'Ram' not in df.columns or df['Ram'].isnull().any():
            INPUT_DATA_MISSING.inc()
        if 'Gpu' in df.columns and 'Nvidia' in df['Gpu'].iloc[0]:
            GPU_REQUESTED_COUNT.inc()

        # Update Metrik Lanjutan
        SCREEN_SIZE_AVG.observe(df['Inches'].iloc[0])
        RAM_USED_AVG.observe(df['Ram'].iloc[0])
        
        # Prediksi dan Catat Waktu Prediksi
        start_pred_time = time.time()
        prediction = model.predict(df)
        end_pred_time = time.time()
        
        PREDICTION_TIME.observe(end_pred_time - start_pred_time)
        PREDICTION_COUNT.inc()
        
        # Kembalikan hasil prediksi
        return jsonify({
            "predicted_price_in_rupiah": prediction[0],
            "status": "success"
        })

    except Exception as e:
        PREDICTION_FAILURE_COUNT.inc()
        return jsonify({"error": str(e)}), 500