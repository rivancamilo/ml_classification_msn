# Importar dependencias desde el módulo centralizado
from librerias import (
    os, pickle, joblib, logging, np, pd, plt,
    Dict, Tuple, Optional, Any,
    CountVectorizer, TfidfTransformer, train_test_split,
    accuracy_score, classification_report, confusion_matrix, 
    ConfusionMatrixDisplay, precision_recall_fscore_support, roc_auc_score,
    LogisticRegression, mlflow, infer_signature
)

class ModelTrain:
   
    def __init__(self, data_processed_path: str = "./data/modelos"):
        self.data_processed_path = data_processed_path
        self.idx2label = {'positivo': 0, 'negativo': 1, 'neutral': 2}
        self.label2idx = {v: k for k, v in self.idx2label.items()}
        self.count_vectorizer = None
        self.tfidf_transformer = None
        self.logger = logging.getLogger(__name__)
        
        # Create data directory if it doesn't exist
        os.makedirs(self.data_processed_path, exist_ok=True)
    
    def data_transform(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        df = df.dropna(subset=['textCls', 'sentimiento'])
        X = df['textCls']
        y = df['sentimiento']
        return X, y
    
    def decode_labels_into_idx(self, labels: pd.Series) -> pd.Series:
        return labels.map(self.idx2label)
    
    def fit_transform(self, X: pd.Series) -> np.ndarray:
        self.count_vectorizer = CountVectorizer()
        X_vectorized = self.count_vectorizer.fit_transform(X)
        # Save count vectorizer for data preprocessing in the main app (deploy)
        joblib.dump(self.count_vectorizer, 
                   os.path.join(self.data_processed_path, 'count_vectorizer.pkl'))
        self.logger.info("Count vectorizer trained successfully stored")
        return X_vectorized
    
    def transform_tfidf(self, X_vectorized: object) -> np.ndarray:
        self.tfidf_transformer = TfidfTransformer()
        X_tfidf = self.tfidf_transformer.fit_transform(X_vectorized)
        joblib.dump(self.tfidf_transformer, 
                   os.path.join(self.data_processed_path, 'tfidf_transformer.pkl'))
        joblib.dump(X_tfidf, 
                   os.path.join(self.data_processed_path, 'X_tfidf.pkl'))
        self.logger.info("TF-IDF transformer and X_tfidf trained successfully stored")
        return X_tfidf
    
    def save_pickle(self, data: Any, filename: str) -> None:
        filepath = os.path.join(self.data_processed_path, f"{filename}.pkl")
        with open(filepath, 'wb') as file:
            pickle.dump(data, file)
    
    def split_train_test(
        self, X_tfidf: np.ndarray, y: pd.Series, 
        test_size: float = 0.3, random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series]:
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_tfidf, y, test_size=test_size, random_state=random_state
        )
        self.save_pickle((X_train, y_train), "train")
        self.save_pickle((X_test, y_test), "test")
        self.logger.info("Data saved successfully in pickle files")
        return X_train, X_test, y_train, y_test
    
    def display_classification_report(
        self,
        model: object,
        name_model: str,
        developer: str,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray
    ) -> list:
        
        metric = []
        y_train_pred_proba = model.predict_proba(X_train)
        y_test_pred_proba = model.predict_proba(X_test)
        
        roc_auc_score_train = round(
            roc_auc_score(y_train, y_train_pred_proba, average="weighted", multi_class="ovr"),2,)
        
        roc_auc_score_test = round(
            roc_auc_score(y_test, y_test_pred_proba, average="weighted", multi_class="ovr"),2,)


        self.logger.info(f"ROC AUC Score Train: {roc_auc_score_train}")
        self.logger.info(f"ROC AUC Score Test: {roc_auc_score_test}")
        
        # Adding the metrics to the list
        metric.extend([roc_auc_score_train, roc_auc_score_test])

        mlflow.log_metric("roc_auc_train", roc_auc_score_train)
        mlflow.log_metric("roc_auc_test", roc_auc_score_test)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        (precision_train,recall_train,fscore_train,support_train,) = precision_recall_fscore_support(
                                                                        y_train, y_train_pred, average="weighted")
        
        (precision_test,recall_test,fscore_test,support_test,) = precision_recall_fscore_support(
                                                                        y_test, y_test_pred, average="weighted")

        mlflow.log_metric("precision_train", precision_train)
        mlflow.log_metric("precision_test", precision_test)
        mlflow.log_metric("recall_train", recall_train)
        mlflow.log_metric("recall_test", recall_test)
        
        signature = infer_signature(X_test, y_test_pred)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=f"model_{name_model}",
            input_example=X_test[:5],
            signature=signature
        )

        acc_score_train = round(accuracy_score(y_train, y_train_pred), 2)
        acc_score_test = round(accuracy_score(y_test, y_test_pred), 2)

        metric.extend(
            [
                acc_score_train,
                acc_score_test,
                round(precision_train, 2),
                round(precision_test, 2),
                round(recall_train, 2),
                round(recall_test, 2),
                round(fscore_train, 2),
                round(fscore_test, 2),
            ]
        )

        print("Train Accuracy: ", acc_score_train)
        print("Test Accuracy: ", acc_score_test)

        model_report_train = classification_report(y_train, y_train_pred)
        model_report_test = classification_report(y_test, y_test_pred)

        print("Classification Report for Train:\n", model_report_train)
        print("Classification Report for Test:\n", model_report_test)

        # Plot the confusion matrix
        """ label_map = self.label2idx
        fig, ax = plt.subplots(figsize=(12, 8))
        decoded_y_test_pred = [label_map[idx] for idx in y_test_pred]
        decoded_y_test = [label_map[idx] for idx in y_test]

        cm = confusion_matrix(decoded_y_test, decoded_y_test_pred)
        cmp = ConfusionMatrixDisplay(cm, display_labels=list(label_map.values()))
        cmp.plot(ax=ax)

        plt.xticks(rotation=80)
        plt.show() """

        # Log model to MLflow again with a different path
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=f"models_{name_model}",
            input_example=X_test[:5],
            signature=signature
        )

        return metric
    
    def train_model(self, model_type: str = "logistic_regression", **kwargs) -> object:
        
        if model_type == "logistic_regression":
            params = {
                'solver': 'saga',
                'C': 1.0,
                'max_iter': 1000,
                'n_jobs': -1
            }
            # Update with any user-provided parameters
            params.update(kwargs)
            model = LogisticRegression(**params)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
            
        return model
    
    def run(self, df: pd.DataFrame, model_type: str = "logistic_regression", 
            developer: str = "Ivan Camilo", **kwargs) -> object:
        
        self.logger.info(f"Starting model training process with {model_type}")
        self.logger.info(f"DataFrame shape: {df.shape}")                                                                      
        X, y = self.data_transform(df)
        self.logger.info(f"After data transform - X shape: {X.shape}, y shape: {y.shape}")                                                                                    
        Y = self.decode_labels_into_idx(labels=y)
        
        # Feature extraction
        X_vectorized = self.fit_transform(X.values)
        self.logger.info(f"After vectorization - X shape: {X_vectorized.shape}")                                                                        
        X_tfidf = self.transform_tfidf(X_vectorized)
        self.logger.info(f"After TFIDF - X shape: {X_tfidf.shape}")                                                           
        
        # Train-test split
        X_train, X_test, y_train, y_test = self.split_train_test(X_tfidf, Y)
        self.logger.info(f"After train-test split - X_train: {X_train.shape}, X_test: {X_test.shape}")                                                                                              
        
        # Model training
        model = self.train_model(model_type=model_type, **kwargs)
        self.logger.info("Fitting model...")                                    
        model.fit(X_train, y_train)
        self.logger.info("Model fitting completed")                                           
        
        # Model evaluation
        metrics = self.display_classification_report(
            model=model,
            name_model=model_type.capitalize(),
            developer=developer,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test
        )
        
        return model
    
"""     
if __name__ == "__main__":
    datos = pd.read_csv('./data/output/feature_twcs_2.csv')
    model_trainer = ModelTrain()
    model = model_trainer.run(
        df=datos,
        model_type="logistic_regression",
        developer="Ivan Camilo Rosales",
        C=1.0,
        max_iter=1000
    ) """