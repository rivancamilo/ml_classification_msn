#!/bin/bash

# Asegurarse de que los directorios críticos existan
mkdir -p /app/mlruns /app/data/input /app/data/output /app/data/modelos
chmod -R 777 /app/mlruns /app/data

export MLFLOW_TRACKING_URI=http://0.0.0.0:5000

# Iniciar MLFlow en segundo plano
echo "Iniciando MLflow UI..."
mlflow server \
    --backend-store-uri sqlite:////app/mlruns/mlflow.db \
    --default-artifact-root /app/mlruns \
    --host 0.0.0.0 &

# Iniciar Prefect en segundo plano
echo "Iniciando Prefect Server..."
prefect server start --host 0.0.0.0 &

# Tiempo de espera para que los servidores estén disponibles
echo "Esperando que los servidores estén listos..."
sleep 10


# Ejecución del experimento
echo "Iniciando ejecución de main.py..."
python main.py

# Mantener el contenedor en ejecución
echo "Servicios iniciados correctamente. Manteniendo el contenedor en ejecución..."
tail -f /dev/null