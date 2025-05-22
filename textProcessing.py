""" 
    textProcessing.py
    Descripción:
        Esta clase se encarga del preprocesamiento de los mensajes de texto, realizando tareas como la eliminación de caracteres especiales, emojis, URLs, dígitos, y otras transformaciones necesarias para preparar los datos antes de entrenar un modelo de machine learning.

    Métodos:
        eliminar_urls(text):
            Elimina las URLs presentes en el texto del mensaje.
        delete_caracter_especial(text):
            Limpia caracteres especiales, incluyendo letras con tildes y otros signos no alfabéticos.
        remove_emoji(text):
            Elimina los emojis presentes en el texto.
        delete_digitos(text):
            Elimina todos los dígitos numéricos del texto.
        delete_puntuacion(text):
            Elimina los signos de puntuación del texto.
        tokenize(text):
            Divide el texto en una lista de palabras o tokens, lo que facilita
            su análisis y procesamiento posterior.
        remove_stopwords(tokens):
            Elimina las palabras vacías (stopwords), es decir, palabras comunes que no 
            aportan información útil al análisis (como "y", "de", "el").
        lemmatize(tokens):
            Convierte cada palabra a su forma base o "lema", ayudando a reducir la 
            variabilidad lingüística (por ejemplo, "corriendo" → "correr").
        procesador_texto(text):
            Aplica de forma secuencial todos los métodos de limpieza y 
            transformación del texto, como la eliminación de emojis, stopwords, dígitos, etc.
        save_processed_data(df, ruta):
            Guarda el DataFrame procesado en un archivo CSV.
        read_csv(ruta):
            Lee un archivo CSV y carga los datos en un DataFrame de pandas.
        data_transform(df):
            Convierte tipos de datos y elimina columnas que no son relevantes para el análisis.
        run():
            Integra y ejecuta todos los métodos de limpieza y transformación para preparar el conjunto de datos final.
    Autor: Ivan Camilo Rosales
    Fecha: 2025-05-21
"""



from librerias import (
    pd, os, re, string, logging, datetime,
    punctuation, word_tokenize, stopwords, SnowballStemmer,
    nltk, WordNetLemmatizer, SentimentIntensityAnalyzer,
    TfidfVectorizer, emoji, plt, WordCloud, STOPWORDS, ImageColorGenerator
)

from config import *


class TextProcessing:
    
    def __init__(self,idioma: str):
        nltk.download('stopwords')
        nltk.download('punkt')
        nltk.download('punkt_tab')
        nltk.download('wordnet')
        nltk.download('omw-1.4')
        nltk.download('vader_lexicon')

        self.idioma = idioma
        self.stemmer = SnowballStemmer(self.idioma)
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words(self.idioma))
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)


    def eliminar_urls(self,texto:str):
        patron = r'https?://\S+|www\.\S+'
        return re.sub(patron, '', texto)
    
    def delete_caracter_especial(self,texto:str):
        return re.sub(r'[^A-Za-záéíóúÁÉÍÓÚñÑ ]+', '', texto)

    def remove_emoji(self, texto:str)-> str:
        textobase = re.sub(r'[\r\n]+', ' ', texto)
        return emoji.replace_emoji(textobase, replace='')

    def delete_digitos(self, texto:str)-> str:
        return ''.join(c for c in texto if not c.isdigit())

    def delete_puntuacion(self, texto:str)-> str:
        return ''.join(c for c in texto if c not in punctuation)
    
    
    def tokenize(self,texto: str)-> str:
        tokens = word_tokenize(texto.lower(), language=self.idioma)
        return tokens
    
    def remove_stopwords(self,tokens: list):
        filtered_tokens = [
            word for word in tokens if word.lower() not in self.stop_words
        ]
        return filtered_tokens  
    
    
    def lemmatize(self,tokens: list):
        lemmatized_tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
        return lemmatized_tokens
    
    
    def procesador_texto(self,columna_df :pd.Series):
        inicio_time = datetime.datetime.now()
        deleteUrls = columna_df.apply(self.eliminar_urls)
        deleteCaracterres = deleteUrls.apply(self.delete_caracter_especial)
        text_sin_emojin = deleteCaracterres.apply(self.remove_emoji)
        text_sin_digitos = text_sin_emojin.apply(self.delete_digitos)
        text_sin_puntuacion = text_sin_digitos.apply(self.delete_puntuacion)
        tokenized_text = text_sin_puntuacion.apply(self.tokenize)
        
        stopwords_text = tokenized_text.apply(self.remove_stopwords)
        lemmatize_text = stopwords_text.apply(self.lemmatize)
        
        fin_time = datetime.datetime.now()
        self.logger.info(f"Preprocesamiento del texto completado")
        self.logger.info(f"Tiempo de Ejecucion: {fin_time - inicio_time}")
        return lemmatize_text
    
    def save_processed_data(self, df: pd.DataFrame, path: str, file_name: str) -> None:
        file_path = os.path.join(path, file_name)
        df.to_csv(file_path, index=False)
        #print(f"Guardado exitoso de datos preprocesados \n\t {file_path}")
        self.logger.info(f"Guardado exitoso de datos preprocesados \n\t {file_path}")
        
    def read_csv(self, path: str, filename: str):
        file_path = os.path.join(path, filename)
        df = pd.read_csv(file_path)
        self.logger.info(f"Ingesta de datos en curso \n\t {file_path}")
        return df
    
    def data_transform(self, df: pd.DataFrame):
        df.drop(["tweet_id"], axis=1, inplace=True)
        df.drop(["author_id"], axis=1, inplace=True)
        df.drop(["created_at"], axis=1, inplace=True)
        df.drop(["response_tweet_id"], axis=1, inplace=True)
        df.drop(["in_response_to_tweet_id"], axis=1, inplace=True)
        
        #-----  convertimos la columna inbound a boolean
        df['inbound'] = df['inbound'].astype(bool)
        #-----  convertimos la columna text a tipo string
        df['text'] = df['text'].astype(str)
        df = df[df['inbound']==True]
        
        df = df.reset_index(drop=True)
        self.logger.info("Transformación básica de datos completada")
        return df
    
    def clsTexto(self,words: list):
        return " ".join(str(word) for word in words)
    
    def run(self,file_name: str, version: int):
        name_data_input = f"{file_name}.csv"
        data = self.read_csv(
            DATA_PATH_INPUT, name_data_input
        )
        
        data = self.data_transform(data)
        text_cls = self.procesador_texto(data['text'])

        data['TextPreproc'] = text_cls
        data['textCls'] = data['TextPreproc'].apply(self.clsTexto)
        self.save_processed_data(
            df=data,
            path=DATA_PATH_PROCESSED,
            file_name=f"processing_{file_name}_{version}.csv",
        )


""" if __name__ == "__main__":
    text_processing = TextProcessing(idioma="english")
    text_processing.run(file_name="twcs", version="1") """