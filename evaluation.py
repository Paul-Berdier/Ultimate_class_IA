# evaluation.py
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

class Evaluator:
    """
    Classe utilitaire pour l'évaluation et la visualisation.
    """
    @staticmethod
    def print_classification_report(y_true, y_pred, model_name="Modèle"):
        print(f"--- Rapport de classification ({model_name}) ---")
        print(classification_report(y_true, y_pred))

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, labels, model_name="Modèle"):
        print(f"--- Matrice de confusion ({model_name}) ---")
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f"Matrice de confusion - {model_name}")
        plt.show()
