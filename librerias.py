# ==== IMPORTACIONES ESTÁNDAR DE PYTHON ====
import os
import re
import json
import string
import pickle
import logging
import warnings
import datetime
from typing import Dict, Tuple, Optional, Any, List, Union
from string import punctuation


# ==== IMPORTACIONES DE PROCESAMIENTO DE DATOS ====
import numpy as np
import pandas as pd
import joblib


# ==== IMPORTACIONES DE VISUALIZACIÓN ====
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator


# ==== IMPORTACIONES DE NLP ====
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
import emoji


# ==== IMPORTACIONES DE SCIKIT-LEARN ====
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_fscore_support,
    roc_auc_score,
)


# ==== IMPORTACIONES DE MLFLOW ====
import mlflow
from mlflow.models.signature import infer_signature


# ==== IMPORTACIONES DE PREFECT ====
from prefect import flow, task


# ==== CONFIGURACIÓN INICIAL ====
def setup_nltk():
    """Configura las dependencias de NLTK necesarias."""
    nltk_packages = [
        'stopwords',
        'punkt',
        'punkt_tab',
        'wordnet',
        'omw-1.4',
        'vader_lexicon'
    ]
    for package in nltk_packages:
        try:
            nltk.download(package, quiet=True)
        except Exception as e:
            logging.warning(f"Error al descargar {package}: {e}")


def setup_logging(level=logging.INFO):
    """Configura el sistema de logging para el proyecto."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


def setup_warnings():
    """Configura los warnings para el proyecto."""
    warnings.filterwarnings("ignore")


def setup_mlflow(tracking_uri, experiment_name):
    """Configura MLflow para el proyecto."""
    
    mlflow.set_tracking_uri(tracking_uri)
    
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
            logger.info(f"Created new experiment '{experiment_name}' with ID: {experiment_id}")
        else:
            experiment_id = experiment.experiment_id
            logger.info(f"Using existing experiment '{experiment_name}' with ID: {experiment_id}")
        
        mlflow.set_experiment(experiment_name)
        return experiment_id
    except Exception as e:
        logger.error(f"Error setting up MLflow experiment: {e}")
        return 0