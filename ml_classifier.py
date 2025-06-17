# ml_classifier.py
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV

class MLTextClassifier:
    """
    Classe Machine Learning pour la classification de texte.
    """
    def __init__(self):
        print("Initialisation du classificateur ML...")
        self.pipeline = None
        self.best_params = None

    def build_pipeline(self, model=None):
        print("Construction du pipeline ML (TF-IDF + LogisticRegression)...")
        if model is None:
            model = LogisticRegression(max_iter=1000)
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('clf', model)
        ])

    def train(self, X, y, optimize=False):
        print("Démarrage de l'entraînement ML...")
        if not self.pipeline:
            self.build_pipeline()
        if optimize:
            print("Optimisation des hyperparamètres via GridSearchCV...")
            params = {
                'tfidf__ngram_range': [(1,1), (1,2)],
                'clf__C': [0.01, 0.1, 1, 10]
            }
            grid = GridSearchCV(self.pipeline, params, cv=3, verbose=1, n_jobs=-1)
            grid.fit(X, y)
            self.pipeline = grid.best_estimator_
            self.best_params = grid.best_params_
            print(f"Meilleurs hyperparamètres : {self.best_params}")
        else:
            self.pipeline.fit(X, y)
        print("Entraînement ML terminé.")

    def predict(self, X):
        print("Prédiction ML en cours...")
        return self.pipeline.predict(X)

    def evaluate(self, X, y):
        print("Évaluation du modèle ML...")
        y_pred = self.predict(X)
        print("Rapport de classification ML :")
        print(classification_report(y, y_pred))
        print("Matrice de confusion :")
        print(confusion_matrix(y, y_pred))
