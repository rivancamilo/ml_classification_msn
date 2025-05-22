# 🧠 Modelo de Clasificación de Tuits 

## Descripción del Proyecto

Sistema de Machine Learning para la clasificación automática de tweets utilizando **MLflow** y **Prefect** como herramientas de orquestación y seguimiento de experimentos.

## 📌 Objetivo

Desarrollar un sistema de clasificación inteligente para tweets recibidos en el canal de soporte técnico, con el propósito de:

- Identificar automáticamente mensajes relacionados con problemas específicos
- Analizar el sentimiento de los mensajes de los usuarios
- Priorizar la atención al cliente basada en el análisis de sentimientos

### Categorías de Clasificación

El modelo clasifica los mensajes en tres categorías principales:

| Categoría | Emoji | Descripción |
|-----------|-------|-------------|
| **Positivo** | ✅ | Mensajes con sentimiento favorable |
| **Neutro** | 😐 | Mensajes informativos sin carga emocional |
| **Negativo** | ❌ | Mensajes que requieren atención prioritaria |


## 🛠️ Tecnologías Utilizadas

- **Python** - Lenguaje principal
- **MLflow** - Seguimiento de experimentos y gestión de modelos
- **Prefect** - Orquestación de flujos de trabajo
- **Docker** - Containerización y despliegue
- **Scikit-learn** - Algoritmos de Machine Learning


## 📋 Requisitos Previos

Antes de ejecutar el proyecto, asegúrate de tener instalado:

- **Git**
- **Docker**

### Verificación de Dependencias

```bash
# Verificar Git
git --version

# Verificar Docker
docker --version

```

## 🚀 Instalación y Ejecución

### 1️⃣ Clonar el Repositorio

```bash
git clone https://github.com/rivancamilo/ml_classification_msn.git

cd ml_classification_msn

```

### 2️⃣ Construir la Imagen Docker

```bash
docker build -t clasificacionmsn .
```

### 3️⃣ Ejecutar el Contenedor

```bash
docker run --name clasificacionmsn_container -p 5000:5000 -p 4200:4200 -v $(pwd)/mlruns:/app/mlruns clasificacionmsn

```


### 4️⃣ Verificar el Estado del Contenedor

El sistema estará listo cuando veas mensajes similares a:

```
 ___ ___ ___ ___ ___ ___ _____
| _ \ _ \ __| __| __/ __|_   _|
|  _/   / _|| _|| _| (__  | |
|_| |_|_\___|_| |___\___| |_|

Configure Prefect to communicate with the server with:

    prefect config set PREFECT_API_URL=http://0.0.0.0:4200/api

View the API reference documentation at http://0.0.0.0:4200/docs

Check out the dashboard at http://0.0.0.0:4200
INFO  [alembic.runtime.migration] Context impl SQLiteImpl.
INFO  [alembic.runtime.migration] Will assume non-transactional DDL.
[INFO] Starting gunicorn 23.0.0
[INFO] Listening at: http://0.0.0.0:5000 (4898)
[INFO] Using worker: sync
[INFO] Booting worker with pid: 4899
Iniciando ejecución de main.py...
[INFO] prefect - Starting temporary server on http://127.0.0.1:8564
```

## 🌐 Acceso a los Servicios

Una vez que el contenedor esté ejecutándose, podrás acceder a:

### 📊 MLflow UI
- **URL:** http://localhost:5000
- **Descripción:** Interfaz para el seguimiento de experimentos, métricas y gestión de modelos


### ⚙️ Prefect UI
- **URL:** http://localhost:4200
- **Descripción:** Dashboard para la orquestación y monitoreo de flujos de trabajo

## 📁 Estructura del Proyecto

```
ml_classification_msn/
├── data/
│   ├── imput/             # DataSet inicial sin preprocesamiento
|   |   ├── twcs.zip       
|   |   ├── twcs.z01
|   |   ├── twcs.z02
│   ├── output/            # DataSet preprocesados
│   ├── modelos/           # Definición y entrenamiento de modelos
├── notebooks/             # Jupyter notebooks de análisis exploratorio de datos y comparación de modelos 
├── librerias.py           # Modulo central para manejar los import de las librerías
├── config.py              # Modulo Central para la configuración de variables globales
├── main.py                
├── textProcessing.py                
├── featureExtraction.py   
├── modelo.py
├── requirements.txt       # Dependencias de Python
├── dockerfile
├── docker-compose.yml
├── entrypoint.sh
└── README.md
```

## 🔧 Comandos Útiles

### Gestión del Contenedor

```bash
# Detener el contenedor
docker stop clasificacionmsn_container

# Reiniciar el contenedor
docker restart clasificacionmsn_container

# Eliminar el contenedor
docker rm clasificacionmsn_container

# Ver logs del contenedor
docker logs clasificacionmsn_container
```

### Acceso al Contenedor

```bash
# Ejecutar bash dentro del contenedor
docker exec -it clasificacionmsn_container bash

```

## 👥 Autores

- **Ivan Camilo Rosales R.** - [@rivancamilo](https://github.com/rivancamilo)
