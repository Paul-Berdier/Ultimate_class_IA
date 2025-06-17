# dl_classifier.py
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

class DLTextClassifier:
    """
    Classe Deep Learning pour la classification de texte (LSTM).
    """
    def __init__(self, num_words=5000, max_len=200):
        print("Initialisation du classificateur DL (LSTM)...")
        self.tokenizer = Tokenizer(num_words=num_words, oov_token="<OOV>")
        self.max_len = max_len
        self.model = None
        self.label_encoder = LabelEncoder()
        self.num_classes = None

    def prepare_data(self, X, y=None, fit_tokenizer=False):
        if fit_tokenizer:
            print("Apprentissage du tokenizer sur les textes...")
            self.tokenizer.fit_on_texts(X)
        print("Transformation des textes en séquences...")
        X_seq = self.tokenizer.texts_to_sequences(X)
        X_pad = pad_sequences(X_seq, maxlen=self.max_len, padding='post')
        if y is not None:
            print("Encodage des labels...")
            y_enc = self.label_encoder.fit_transform(y)
            self.num_classes = len(np.unique(y_enc))
            y_cat = to_categorical(y_enc, num_classes=self.num_classes)
            return X_pad, y_cat
        return X_pad

    def build_model(self, embedding_dim=64, lstm_units=64, dropout=0.2):
        print("Construction du modèle LSTM...")
        self.model = Sequential([
            Embedding(input_dim=len(self.tokenizer.word_index)+1, output_dim=embedding_dim, input_length=self.max_len),
            LSTM(lstm_units),
            Dropout(dropout),
            Dense(self.num_classes, activation='softmax')
        ])
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(self.model.summary())

    def train(self, X, y, epochs=5, batch_size=32, validation_split=0.1):
        print("Entraînement du modèle DL...")
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=validation_split, verbose=2)
        print("Entraînement DL terminé.")

    def predict(self, X):
        print("Prédiction DL en cours...")
        # Transformation du texte en séquences et padding
        X_pad = self.tokenizer.texts_to_sequences(X)
        X_pad = pad_sequences(X_pad, maxlen=self.max_len, padding='post')
        pred = self.model.predict(X_pad)
        y_pred = np.argmax(pred, axis=1)
        return self.label_encoder.inverse_transform(y_pred)

    def evaluate(self, X, y):
        print("Évaluation du modèle DL...")
        score = self.model.evaluate(X, y, verbose=0)
        print(f"Perte (loss) : {score[0]:.4f} | Précision (accuracy) : {score[1]:.4f}")
