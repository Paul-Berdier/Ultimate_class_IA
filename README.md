# Ultimate_class_IA

*Le pipeline ultime pour automatiser le NLP supervisé comme un boss en M1 Data Science (YNOV) !*

---

## 📚 **Contexte**

Ce projet a été réalisé dans le cadre du module **NLP** (Natural Language Processing) du M1 Data Science à YNOV (2024-2025), sous la supervision de **Nicolas Miotto**.  
**Objectif :** automatiser toutes les étapes du prétraitement, de la classification ML/DL et de la prédiction sur du texte… et tout encapsuler en Programmation Orientée Objet (POO) comme des pros.

---

## 🚀 **Fonctionnalités**

- **Prétraitement automatique** des jeux de données texte (nettoyage, normalisation, gestion des doublons/NA…).
- **Classification supervisée** :
  - **Machine Learning** : Pipeline Scikit-learn (TF-IDF + LogisticRegression, GridSearch).
  - **Deep Learning** : LSTM (Keras/TensorFlow), embeddings, fine-tuning express.
- **Évaluation complète** : Rapport de classification, matrice de confusion, print de tous les scores.
- **Pipeline tout-en-un** : Orchestration POO pour automatiser les tests, y compris sur des nouveaux jeux de données.
- **Prédiction sur articles inédits** (Annexe 1 fournie dans le sujet).

---

## 🗂️ **Structure du repo**

```

ultimate_class_ia/
│
├── preprocessing.py       # Classe de prétraitement du texte (Pandas, nettoyage)
├── ml_classifier.py       # Classe ML (Scikit-learn, TF-IDF, LogReg)
├── dl_classifier.py       # Classe Deep Learning (Keras, LSTM)
├── evaluation.py          # Classe utilitaire pour métriques et plots
├── orchestrator.py        # Orchestrateur de pipeline (lance tout)
├── main.py                # Script principal, tout s’enchaîne
├── bbc_dataset.csv        # Dataset texte (prévu pour l’étude)
├── financial_data.csv     # Pour la généralisation (Partie V)
├── requirements.txt       # Dépendances Python
└── README.md              # Ce fichier (beau, utile, complet)

````

---

## 🏁 **Installation**

1. **Clone ce repo :**
    ```bash
    git clone https://github.com/TonPseudo/Ultimate_class_IA.git
    cd Ultimate_class_IA
    ```

2. **Crée un venv et installe les dépendances :**
    ```bash
    python -m venv .venv
    source .venv/bin/activate     # (ou .venv\Scripts\activate sous Windows)
    pip install -r requirements.txt
    ```

3. **Place les fichiers CSV** (`bbc_dataset.csv`, `financial_data.csv`) à la racine du projet.

---

## 🛠️ **Utilisation**

Lance simplement :
```bash
python main.py
````

Tu verras :

* Le pipeline ML s’entraîner, s’évaluer, prédire sur les nouveaux textes.
* Le pipeline DL faire la même chose (LSTM, embeddings…).
* Un affichage très verbeux (full print) pour suivre chaque étape.
* Les prédictions sur les textes de l’annexe.

---

## 📈 **Résultats type**

Exemple de sortie (tu peux le coller dans ton rapport) :

```
--- Pipeline ML ---
Meilleurs hyperparamètres : {'clf__C': 10, 'tfidf__ngram_range': (1, 2)}
Accuracy : 97%
...
--- Pipeline DL ---
Epoch 5/5 - accuracy: 0.73
...
--- Prédiction ML ---
['sport', 'tech', 'sport']
--- Prédiction DL ---
['sport', 'sport', 'sport']
```

---

## 🤔 **À adapter/améliorer**

* Changer d’architecture LSTM (Bidirectional, GRU…).
* Tester d’autres modèles ML (SVM, Naive Bayes).
* Améliorer la gestion du financial\_data.csv pour la généralisation totale.
* Ajouter des plots matplotlib pour les courbes d’apprentissage.

---

## 👨‍💻 **Crédits**

* **Paul Berdier** (M1 Data Science, YNOV)

---

## 🧠 **Disclaimer**

> *“Ce repo est conçu pour être lisible, pédagogique, facilement modifiable, et bourré de prints pour la correction (et la paresse).
> En cas de bug : relis le README, et si ça ne marche toujours pas, c’est probablement un problème de CSV ou d’environnement virtuel.”*

