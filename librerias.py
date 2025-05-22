""" 
    librerias.py
    Descripción:
        Este módulo reúne y gestiona todas las librerías empleadas en el proyecto
    Autor: Ivan Camilo Rosales
    Fecha: 2025-05-21
"""

#-----  Librerías básicas de Python
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


#-----  Librerías para el procesamiento de los datos
import numpy as np
import pandas as pd
import joblib


#-----  Librerías para la generación de graficas
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator


#----   Librerías de NLP
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
import emoji


#-----  Librerías de SCIKIT-LEARN
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


#-----  Librerías de MLFlow
import mlflow
from mlflow.models.signature import infer_signature


#-----  Librerías de Perfect
from prefect import flow, task


#-----  Funciones de configuración inicial
def setup_nltk():
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
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


def setup_warnings():
    warnings.filterwarnings("ignore")
