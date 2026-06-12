from pathlib import Path
import joblib
import numpy as np
from src.preprocessor import TextPreprocessor


class FakeNewsPredictor:

    def __init__(self):
        base_dir = Path(__file__).resolve().parent.parent
        models_dir = base_dir / "models"
        self.preprocessor = TextPreprocessor()
        self.vectorizer = joblib.load(
            models_dir / "tfidf_vectorizer.pkl"
        )
        self.models = {
            "Logistic Regression": joblib.load(
                models_dir / "logistic_regression.pkl"
            ),

            "Naive Bayes": joblib.load(
                models_dir / "naive_bayes.pkl"
            ),

            "Random Forest": joblib.load(
                models_dir / "random_forest.pkl"
            )
        }

    def predict(self, text: str, model_name: str):

        cleaned = self.preprocessor.clean(text)
        vec = self.vectorizer.transform([cleaned])
        model = self.models[model_name]
        prediction = model.predict(vec)[0]
        probabilities = model.predict_proba(vec)[0]

        return {
            "label": "FAKE" if prediction == 0 else "REAL",

            "confidence": float(
                np.max(probabilities)
            ),

            "probabilities": {
                "FAKE": float(probabilities[0]),
                "REAL": float(probabilities[1])
            }
        }