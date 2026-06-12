import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


class TextPreprocessor:
    """
    Handles text cleaning and preprocessing for fake news detection.
    """

    def __init__(self):
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()

    def clean(self, text: str) -> str:
        """
        Clean a single text document.
        """

        text = str(text).lower()

        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)

        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)

        # Remove non-letters
        text = re.sub(r'[^a-z\s]', '', text)

        # Tokenize
        tokens = text.split()

        # Stopword removal + lemmatization
        tokens = [
            self.lemmatizer.lemmatize(word)
            for word in tokens
            if word not in self.stop_words and len(word) > 2
        ]

        return " ".join(tokens)

    def batch_clean(self, texts):
        """
        Clean multiple documents.
        """
        return [self.clean(text) for text in texts]