from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier


class ModelTrainer:

    def __init__(self):

        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True
        )

        self.models = {
            "Logistic Regression":
                LogisticRegression(
                    max_iter=1000,
                    random_state=42
                ),

            "Naive Bayes":
                MultinomialNB(alpha=0.1),

            "Random Forest":
                RandomForestClassifier(
                    n_estimators=100,
                    random_state=42,
                    n_jobs=-1
                )
        }