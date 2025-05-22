""" 
    Descripción:
        Módulo principal que coordina el flujo de trabajo del proyecto, 
        desde la ingesta, limpieza y transformación de los datos, hasta 
        el entrenamiento del modelo encargado de clasificar los tuits. Todo 
        este proceso se realiza mediante tareas (tasks).
        
    Autor: Ivan Camilo Rosales
    Fecha: 2025-05-21
"""


# Importar las dependencias centralizadas
from librerias import (
    os, np, pd, mlflow, flow, task, logging, warnings,
    setup_logging, setup_warnings, setup_mlflow
)

from config import *
from featureExtraction import FeatureExtraction
from textProcessing import TextProcessing
from modelo import ModelTrain
from config import MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME




logger = setup_logging()
setup_warnings()

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-----  Validamo si el URI de seguimiento de MLflow está configurado en el entorno, en caso contrario lo configuramos
if not os.environ.get("MLFLOW_TRACKING_URI"):
    os.environ["MLFLOW_TRACKING_URI"] = "sqlite:///mlflow.db"

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-----  Creamos un experimento si no existe
try:
    experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if experiment is None:
        experiment_id = mlflow.create_experiment(MLFLOW_EXPERIMENT_NAME)
        logger.info(f"Created new experiment '{MLFLOW_EXPERIMENT_NAME}' with ID: {experiment_id}")
    else:
        experiment_id = experiment.experiment_id
        logger.info(f"Using existing experiment '{MLFLOW_EXPERIMENT_NAME}' with ID: {experiment_id}")
except Exception as e:
    logger.error(f"Error setting up MLflow experiment: {e}")
    experiment_id = 0

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-----  Establecemos el experimento de MLFLOW
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)



#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-----  Creamos la primera tarea 
@task(retries=2, retry_delay_seconds=2,
      name="Task preprocesamiento de texto", 
      tags=["limpieza_texto"])
def text_processing_task(idioma: str, file_name: str, version: int):
    
    logger.info(f"Iniciamos la tarea de preprocesamiento de texto - file_name={file_name}, version={version}")
    text_processing = TextProcessing(idioma=idioma)
    text_processing.run(file_name=file_name, version=version)
    logger.info("Tarea de procesamiento de texto completada")


#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-----  Creamos la segunda tarea
@task(retries=2, retry_delay_seconds=2,
      name="Tarea de extracción de Features", 
      tags=["extraccion_feature", "categorizacion_texto"])
def feature_extraccion(file_name: str, version: int):
    logger.info(f"Iniciamos la tarea de extracción de features - file_name={file_name}, version={version}")
    feature_extraction = FeatureExtraction()
    feature_extraction.run(file_name=file_name, version=version)
    logger.info("Tarea de extracción de Features completada")


#-------------------------------------------------------------------------------    
#-------------------------------------------------------------------------------   
#----- Creamos la tercera tarea 
@task(
    retries=1,
    name="Entrenamiento del mejor modelo",
    tags=["train", "mejor_modelo", "LogisticRegressionClassifier"],
)
def training_model(file_name: str = FILE_NAME_DATA_FEATURE, version: int = VERSION):
    logger.info(f"Iniciamos la tarea de entrenamiento del modelo - file_name={file_name}, version={version}")
    name_model = "logistic_regression"
    developer = DEVELOPER_NAME
    
    with mlflow.start_run(run_name=name_model) as run:
        run_id = run.info.run_id
        logger.info(f"Se inicio la ejecución de MLFLOW con ID: {run_id}")
        
        #-----  Parámetros básicos del registro
        mlflow.log_param("model", name_model)
        mlflow.log_param("developer", developer)
        mlflow.log_param("file_name", file_name)
        mlflow.log_param("version", version)
        
        #-----  Cargamos los datos
        data_path = f"{DATA_PATH_PROCESSED}/{file_name}_{version}.csv"
        logger.info(f"Loading data from {data_path}")
        datos = pd.read_csv(data_path)
        
        #-----  Entrenamos el modelo
        model_trainer = ModelTrain()
        model = model_trainer.run(
            df=datos,
            model_type="logistic_regression",
            developer=developer,
            C=PARAMETERS_MODEL.get("C", 1.0),
            max_iter=PARAMETERS_MODEL.get("max_iter", 1000),
            random_state=PARAMETERS_MODEL.get("random_state", 40),
            solver=PARAMETERS_MODEL.get("solver", "liblinear")
        )
        
        logger.info(f"Entrenamiento del modelo completado correctamente. ID de ejecución: {run_id}")
        return model

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-----  Creación del flujo principal
@flow(name="Pipeline Clasificación de Tuits")
def main_flow():
    logger.info("Inicio del flujo principal")
    text_processing_task(idioma=IDIOMA, file_name=FILE_NAME_DATA_INPUT, version=VERSION)
    feature_extraccion(file_name=FILE_NAME_DATA_INPUT, version=VERSION)
    model = training_model(file_name=FILE_NAME_DATA_FEATURE, version=VERSION)
    logger.info("Flujo principal completado exitosamente")
    return model


#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-----  Llamado del método main
if __name__ == "__main__":
    try:
        logger.info("Iniciando la aplicación")
        main_flow()
        logger.info("Solicitud completada exitosamente")
    except Exception as e:
        logger.error(f"Se produjo un error durante la ejecución: {e}", exc_info=True)