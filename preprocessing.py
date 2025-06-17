# preprocessing.py
import pandas as pd
import numpy as np
import string

class DataPreprocessor:
    """
    Classe pour nettoyer, normaliser et transformer les textes.
    """
    def __init__(self, df, text_col, label_col=None):
        print("Initialisation du préprocesseur...")
        self.df = df.copy()
        self.text_col = text_col
        self.label_col = label_col
        self.clean_texts = []
        self.labels = []
        self._clean_data()

    def _clean_text(self, text):
        if pd.isna(text):
            return ""
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = ' '.join(text.split())
        return text

    def _clean_data(self):
        print("Nettoyage des données (doublons, valeurs manquantes, formatage)...")
        self.df = self.df.drop_duplicates(subset=[self.text_col])
        self.df = self.df.dropna(subset=[self.text_col])
        self.df[self.text_col] = self.df[self.text_col].apply(self._clean_text)
        self.clean_texts = self.df[self.text_col].tolist()
        if self.label_col:
            self.labels = self.df[self.label_col].tolist()
        print(f"Nombre de textes nettoyés: {len(self.clean_texts)}")

    def get_clean_texts(self):
        return self.clean_texts

    def get_labels(self):
        return self.labels

    def get_dataframe(self):
        return self.df
