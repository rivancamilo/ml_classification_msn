
""" 
    config.py
    Descripción:
        En este moduo se configuran todas las variables globales del proyecto
    Autor: Ivan Camilo Rosales
    Fecha: 2025-05-21

"""

#-----  Path de los archivos input/output
DATA_PATH_INPUT = "./data/input"
DATA_PATH_PROCESSED = "./data/output"
MODELOS_PATH = "./data/modelos"

#-----  Variables de entorno para MLFlow
MLFLOW_TRACKING_URI = "http://0.0.0.0:5000"
MLFLOW_EXPERIMENT_NAME = "Clasificación_Tuits"

#-----  Versión del desarrollo del modelo
VERSION = 2

#-----  Idioma en que están los datos
IDIOMA = "english"

#-----  Nombre del archivo inicial
FILE_NAME_DATA_INPUT = "twcs"

#----- Nombre del archivo que tiene la data lista para ser procesada por el modelo
FILE_NAME_DATA_FEATURE = "feature_twcs"

#-----  Parámetros del Modelo
PARAMETERS_MODEL = {
    "C": 1.0,
    "class_weight": None,
    "l1_ratio": None,
    "max_iter": 100,
    "penalty": "l2",
    "random_state": 40,
    "solver": "liblinear",
    "tol": 0.0001,
}

#-----  Variables generales
DEVELOPER_NAME = "Ivan Camilo Rosales"
MODEL_NAME = "LogisticRegression"