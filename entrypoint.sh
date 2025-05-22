#!/bin/bash

# entrypoint.sh
# Descripción: 
#   Shell, encargada de validar la creación de las carpetas más importantes del proyecto, 
#   y de iniciar los servicios de MLflow y Prefect
# Autor: Ivan Camilo Rosales R.
# Fecha: 2025-05-21
# Version: 1.0




#-----  Validamos que todos los directorios más importantes estén creados
mkdir -p /app/mlruns /app/data/input /app/data/output /app/data/modelos
chmod -R 777 /app/mlruns /app/data

export MLFLOW_TRACKING_URI=http://0.0.0.0:5000

#-----  Ejecutamos el comando base para iniciar el servicio de MLFlow
echo "**********************************************************************"
echo "**********************************************************************"
echo "**********************************************************************"
echo "Iniciando MLflow UI..."
mlflow server \
    --backend-store-uri sqlite:////app/mlruns/mlflow.db \
    --default-artifact-root /app/mlruns \
    --host 0.0.0.0 &

#-----  Ejecutamos el comando base para iniciar el servicio de Perfect
echo "**********************************************************************"
echo "**********************************************************************"
echo "**********************************************************************"
echo "Iniciando Prefect Server..."
prefect server start --host 0.0.0.0 &

#-----  Tiempo de espera para que los servicios de MlFlow y Perfect estén disponibles
echo "Esperando que los servidores estén listos..."
sleep 25


#-----  Ejecutamos el comando el archivo inicial de Python
echo "Iniciando ejecución de main.py..."
python main.py

#-----  Comando para mantener el contenedor en ejecución
echo "Servicios iniciados correctamente. Manteniendo el contenedor en ejecución..."
tail -f /dev/null