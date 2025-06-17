# orchestrator.py
import pandas as pd
from preprocessing import DataPreprocessor
from ml_classifier import MLTextClassifier
from dl_classifier import DLTextClassifier

class PredictionOrchestrator:
    """
    Classe centrale pour orchestrer tout le pipeline.
    """
    def __init__(self, dataset_path, text_col, label_col):
        print("Chargement du dataset...")
        self.df = pd.read_csv(dataset_path)
        self.preprocessor = DataPreprocessor(self.df, text_col, label_col)
        self.texts = self.preprocessor.get_clean_texts()
        self.labels = self.preprocessor.get_labels()
        print(f"Dataset chargé : {len(self.texts)} lignes.")

    def run_ml(self):
        print("\n--- Pipeline ML ---")
        clf = MLTextClassifier()
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(self.texts, self.labels, test_size=0.2, random_state=42)
        clf.build_pipeline()
        clf.train(X_train, y_train, optimize=True)
        clf.evaluate(X_test, y_test)
        return clf

    def run_dl(self):
        print("\n--- Pipeline DL ---")
        clf = DLTextClassifier()
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(self.texts, self.labels, test_size=0.2, random_state=42)
        X_train_pad, y_train_cat = clf.prepare_data(X_train, y_train, fit_tokenizer=True)
        X_test_pad, y_test_cat = clf.prepare_data(X_test, y_test)
        clf.build_model()
        clf.train(X_train_pad, y_train_cat, epochs=5)
        clf.evaluate(X_test_pad, y_test_cat)
        return clf

    def predict_text(self, model, texts):
        print("\nPrédiction sur nouveaux textes :")
        return model.predict(texts)
