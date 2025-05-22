""" 
    featureExtraction.py
    Descripción:
        Esta clase asigna una de las tres categorías (positivo, negativo, neutral) a cada tuit, 
        basándose en el score obtenido del texto.
        
    Métodos:
        save_processed_data:
            Guarda los datos del DataFrame en un archivo .csv.
        read_csv: 
            Carga el archivo preprocesado y lo convierte en un DataFrame.
        etiquetar_sentimiento: 
            Recibe un texto y le asigna una categoría según el puntaje (score) obtenido.
        data_transform: 
            Recibe un DataFrame y convierte todas las columnas según el tipo de dato que contienen.
        run:   
            Integra todos los métodos anteriores para ejecutar el proceso completo de forma secuencial.
    Autor: Ivan Camilo Rosales
    Fecha: 2025-05-21
    
"""

from librerias import (
    pd, os, logging, 
    SentimentIntensityAnalyzer,
    plt
)
from config import *

class FeatureExtraction:
    
    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.sia = SentimentIntensityAnalyzer()

    def save_processed_data(self, df: pd.DataFrame, path: str, file_name: str) -> None:
        file_path = os.path.join(path, file_name)
        df.to_csv(file_path, index=False)
        self.logger.info(f"Guardado exitoso de datos preprocesados \n\t {file_path}")
        
    def read_csv(self, path: str, filename: str):
        file_path = os.path.join(path, filename)
        df = pd.read_csv(file_path)
        self.logger.info(f"Ingesta de datos en curso \n\t {file_path}")
        return df

    def etiquetar_sentimiento(self,texto:str):
        score = self.sia.polarity_scores(texto)['compound']
        if score >= 0.05:
            return 'positivo'
        elif score <= -0.05:
            return 'negativo'
        else:
            return 'neutral'
        
    def data_transform(self, df: pd.DataFrame):
        #-----  convertimos la columna inbound a boolean
        df['inbound'] = df['inbound'].astype(bool)
        #-----  convertimos la columna text a tipo string
        df['text'] = df['text'].astype(str)
        df['TextPreproc'] = list(df['TextPreproc'])
        df['textCls'] = df['textCls'].astype(str)
        df = df.reset_index(drop=True)
        self.logger.info("Transformación básica de datos completada")
        return df
    
    def run(self,file_name: str, version: int):
        name_data_input = f"processing_{file_name}_{version}.csv"
        data = self.read_csv(
            DATA_PATH_PROCESSED, name_data_input
        )
        
        data = self.data_transform(data)
        data = data.dropna(axis=1)
        data.dropna(inplace=True)
        data['sentimiento'] = data['textCls'].apply(self.etiquetar_sentimiento)
        
        self.save_processed_data(
            df=data,
            path=DATA_PATH_PROCESSED,
            file_name=f"feature_{file_name}_{version}.csv",
        )

""" if __name__ == "__main__":
    text_processing = FeatureExtraction()
    text_processing.run(file_name="twcs", version="2") """