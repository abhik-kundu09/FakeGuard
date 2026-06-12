# PLAN.md — Fake News Detection System
### Academic Portfolio Project | NLP + ML + Streamlit
> **AI Assistant Instruction Manual** — Follow this document phase-by-phase, top to bottom, without skipping. Every section is a contract: complete it fully before moving on. This document is itself part of the academic submission.

---

## Project Identity

| Field | Value |
|---|---|
| **Project Name** | FakeGuard — Fake News Detection System |
| **Core Stack** | Python 3.10+, Scikit-learn, NLTK, Streamlit |
| **Primary Logic** | Jupyter Notebook (`.ipynb`) |
| **Frontend** | Streamlit app (`app.py`) |
| **Vectorization** | TF-IDF |
| **Models** | Logistic Regression, Multinomial Naive Bayes, Random Forest |
| **Submission Type** | Academic + Portfolio |

---

## Final Project Structure (Build Toward This)

```
fake-news-detector/
│
├── notebooks/
│   └── fake_news_detection.ipynb       ← MAIN LOGIC LIVES HERE
│
├── src/
│   ├── __init__.py
│   ├── preprocessor.py                 ← Text cleaning pipeline
│   ├── model_trainer.py                ← Training + evaluation logic
│   └── predictor.py                    ← Inference wrapper
│
├── models/
│   ├── tfidf_vectorizer.pkl
│   ├── logistic_regression.pkl
│   ├── naive_bayes.pkl
│   └── random_forest.pkl
│
├── data/
│   ├── raw/
│   │   ├── Fake.csv
│   │   └── True.csv
│   └── processed/
│       └── cleaned_data.csv
│
├── app.py                              ← Streamlit frontend
├── requirements.txt
├── README.md
└── PLAN.md                             ← This file
```

---

## Dataset

**Source:** Kaggle — "Fake and Real News Dataset"
- URL: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
- Files needed: `Fake.csv`, `True.csv`
- Place both in `data/raw/`
- `Fake.csv` → label `0` (fake)
- `True.csv` → label `1` (real)

**Columns used:** `title`, `text`
**Target column:** `label`

---

---

# PHASE 1 — Environment Setup

**Goal:** Reproducible dev environment, all dependencies pinned.

### Step 1.1 — Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

### Step 1.2 — Create `requirements.txt`
Create this file exactly:
```
numpy==1.26.4
pandas==2.2.1
scikit-learn==1.4.2
nltk==3.8.1
streamlit==1.33.0
matplotlib==3.8.4
seaborn==0.13.2
plotly==5.21.0
joblib==1.4.0
wordcloud==1.9.3
jupyter==1.0.0
ipykernel==6.29.4
```

### Step 1.3 — Install Everything
```bash
pip install -r requirements.txt
python -m nltk.downloader punkt stopwords wordnet averaged_perceptron_tagger
```

### Step 1.4 — Create Folder Structure
```bash
mkdir -p notebooks src models data/raw data/processed
touch src/__init__.py
```

### Phase 1 Checkpoint ✓
- [ ] `pip list` shows all packages
- [ ] NLTK corpora downloaded without error
- [ ] All folders exist

---

---

# PHASE 2 — Data Preparation & EDA (Jupyter Notebook)

**File:** `notebooks/fake_news_detection.ipynb`
**This is the core deliverable notebook. Build everything here first.**

### Step 2.1 — Notebook Cell 1: Imports
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import nltk
import re
import warnings
warnings.filterwarnings('ignore')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
```

### Step 2.2 — Notebook Cell 2: Load Data
```python
fake_df = pd.read_csv('../data/raw/Fake.csv')
true_df = pd.read_csv('../data/raw/True.csv')

fake_df['label'] = 0
true_df['label'] = 1

df = pd.concat([fake_df, true_df], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print(df.shape)
df.head()
```

### Step 2.3 — EDA Cells (Required for Submission)

Build the following visualizations inside the notebook (use Plotly for interactivity):

| Chart | What to Show |
|---|---|
| Class Distribution Bar | Count of fake vs real |
| Article Length Distribution | Histogram of word counts |
| Top 20 Words — Fake | Bar chart from fake articles |
| Top 20 Words — Real | Bar chart from real articles |
| Heatmap | Correlation of numeric features |
| WordCloud (optional bonus) | Most frequent fake news words |

Use markdown cells to explain each chart. Label sections with `## 1. Data Loading`, `## 2. EDA`, etc.

### Step 2.4 — Save Cleaned Data
```python
df[['text', 'label']].to_csv('../data/processed/cleaned_data.csv', index=False)
```

### Phase 2 Checkpoint ✓
- [ ] Notebook runs top to bottom without error
- [ ] All 4+ charts render correctly
- [ ] `cleaned_data.csv` saved to `data/processed/`

---

---

# PHASE 3 — Text Preprocessing Pipeline

**Files:**
- Notebook Section: `## 3. Preprocessing` (build + test here first)
- Then extract into: `src/preprocessor.py`

### Step 3.1 — Build Preprocessor in Notebook

The `clean_text()` function must perform these steps **in order**:

1. **Lowercase** — `text.lower()`
2. **Remove URLs** — regex `https?://\S+`
3. **Remove HTML tags** — regex `<.*?>`
4. **Remove punctuation & special chars** — regex `[^a-z\s]`
5. **Tokenization** — `word_tokenize(text)`
6. **Stopword Removal** — NLTK English stopwords
7. **Lemmatization** — `WordNetLemmatizer().lemmatize(word)`
8. **Rejoin** — `' '.join(tokens)`

```python
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words and len(w) > 2]
    return ' '.join(tokens)
```

Apply to dataframe:
```python
df['cleaned_text'] = df['text'].apply(clean_text)
```

Show a before/after comparison cell in the notebook for 3 sample rows.

### Step 3.2 — Extract to `src/preprocessor.py`

```python
# src/preprocessor.py
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def clean(self, text: str) -> str:
        # [full implementation of clean_text here]
        ...

    def batch_clean(self, texts) -> list:
        return [self.clean(t) for t in texts]
```

### Phase 3 Checkpoint ✓
- [ ] `clean_text()` tested on 5+ examples in notebook
- [ ] `preprocessor.py` importable without error
- [ ] Before/after cell visible in notebook

---

---

# PHASE 4 — TF-IDF Vectorization + Model Training

**Notebook Section:** `## 4. Vectorization & Model Training`
**Then extract to:** `src/model_trainer.py`

### Step 4.1 — TF-IDF Setup in Notebook

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

X = df['cleaned_text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=2)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)
```

**Important:** `fit_transform` on train only. `transform` on test only. Never fit on test.

### Step 4.2 — Train All Three Models

```python
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, C=1.0, random_state=42),
    'Naive Bayes': MultinomialNB(alpha=0.1),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
}

trained_models = {}
for name, model in models.items():
    model.fit(X_train_tfidf, y_train)
    trained_models[name] = model
    print(f"{name} trained.")
```

### Step 4.3 — Evaluation (All Three Required)

For each model compute and display:

| Metric | How |
|---|---|
| Accuracy | `accuracy_score` |
| Precision | `precision_score` |
| Recall | `recall_score` |
| F1-Score | `f1_score` |
| Confusion Matrix | `confusion_matrix` → heatmap via seaborn |
| ROC-AUC | `roc_auc_score` + ROC curve plot |
| Classification Report | `classification_report` |

Build a **comparison DataFrame** and display it as a styled table:
```python
results_df = pd.DataFrame(results).T
results_df.style.highlight_max(color='lightgreen').highlight_min(color='#ffcccc')
```

### Step 4.4 — Save All Artifacts

```python
import joblib

joblib.dump(tfidf, '../models/tfidf_vectorizer.pkl')
joblib.dump(trained_models['Logistic Regression'], '../models/logistic_regression.pkl')
joblib.dump(trained_models['Naive Bayes'], '../models/naive_bayes.pkl')
joblib.dump(trained_models['Random Forest'], '../models/random_forest.pkl')
```

### Step 4.5 — Extract to `src/model_trainer.py`

```python
# src/model_trainer.py
class ModelTrainer:
    def __init__(self, max_features=10000):
        ...
    def train(self, X_train, y_train): ...
    def evaluate(self, X_test, y_test) -> dict: ...
    def save_all(self, output_dir='../models/'): ...
```

### Phase 4 Checkpoint ✓
- [ ] All 3 models trained and evaluated
- [ ] Comparison table rendered in notebook
- [ ] Confusion matrices visible for all 3
- [ ] ROC curves plotted (all 3 on one graph)
- [ ] All 4 `.pkl` files saved to `models/`

---

---

# PHASE 5 — Inference Module

**File:** `src/predictor.py`
**Also test in notebook section:** `## 5. Inference Test`

### Step 5.1 — Build Predictor Class

```python
# src/predictor.py
import joblib
import numpy as np
from src.preprocessor import TextPreprocessor

class FakeNewsPredictor:
    MODEL_PATHS = {
        'Logistic Regression': 'models/logistic_regression.pkl',
        'Naive Bayes':         'models/naive_bayes.pkl',
        'Random Forest':       'models/random_forest.pkl',
    }
    VECTORIZER_PATH = 'models/tfidf_vectorizer.pkl'

    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.vectorizer = joblib.load(self.VECTORIZER_PATH)
        self.models = {
            name: joblib.load(path)
            for name, path in self.MODEL_PATHS.items()
        }

    def predict(self, text: str, model_name: str) -> dict:
        """
        Returns:
            {
                'label': 'FAKE' | 'REAL',
                'confidence': float (0–1),
                'probabilities': {'FAKE': float, 'REAL': float}
            }
        """
        cleaned = self.preprocessor.clean(text)
        vec = self.vectorizer.transform([cleaned])
        model = self.models[model_name]
        prediction = model.predict(vec)[0]
        proba = model.predict_proba(vec)[0]
        return {
            'label': 'FAKE' if prediction == 0 else 'REAL',
            'confidence': float(max(proba)),
            'probabilities': {'FAKE': float(proba[0]), 'REAL': float(proba[1])}
        }
```

### Step 5.2 — Test in Notebook

```python
# In notebook section 5:
import sys; sys.path.append('..')
from src.predictor import FakeNewsPredictor

predictor = FakeNewsPredictor()

sample_fake = "BREAKING: Scientists prove vaccines cause autism, government cover-up exposed"
sample_real = "Federal Reserve raises interest rates by 25 basis points amid inflation concerns"

for text in [sample_fake, sample_real]:
    result = predictor.predict(text, 'Logistic Regression')
    print(f"Text: {text[:60]}...")
    print(f"Prediction: {result['label']} | Confidence: {result['confidence']:.2%}\n")
```

### Phase 5 Checkpoint ✓
- [ ] `FakeNewsPredictor` loads without error
- [ ] `predict()` returns correct dict structure
- [ ] Tested on 3+ sample texts in notebook
- [ ] Inference section visible in notebook

---

---

# PHASE 6 — Streamlit Frontend (`app.py`)

**File:** `app.py` (root level)
**This is the UI layer — all ML logic comes from `src/` imports, never re-implement here.**

### Step 6.1 — UI Design Spec

**Theme:** Dark intelligence — deep navy/charcoal base, sharp red accent for FAKE, sharp green for REAL. Monospace elements for "data feel". Modern, academic credibility.

**Color Tokens:**
```python
# Use these in st.markdown CSS injection
BG_DARK    = "#0D1117"
BG_CARD    = "#161B22"
ACCENT_RED = "#FF4B4B"
ACCENT_GRN = "#00CC88"
TEXT_MAIN  = "#E6EDF3"
TEXT_MUTED = "#8B949E"
BORDER     = "#30363D"
```

**Fonts:** Import via `@import url(...)` in injected CSS
- Display: `Space Grotesk`
- Body: `Inter`
- Mono data labels: `JetBrains Mono`

### Step 6.2 — Page Structure (Build in This Order)

```
app.py layout:
├── [Header] Logo + Title + Subtitle
├── [Sidebar]
│   ├── Model Selector (radio: LR / NB / RF)
│   ├── About section
│   └── Model performance summary (mini table)
│
├── [Main — Tab 1: Detector]
│   ├── Textarea input ("Paste news article here...")
│   ├── Analyze button
│   └── [Results Card — shows after button click]
│       ├── FAKE / REAL verdict banner (colored)
│       ├── Confidence score (large %)
│       ├── Probability bar chart (Plotly)
│       └── Preprocessing preview (expandable)
│
├── [Main — Tab 2: Model Comparison]
│   ├── Metrics table (all 3 models)
│   ├── Bar chart: Accuracy comparison
│   └── Confusion matrix display
│
└── [Footer] — Academic info, tech stack badges
```

### Step 6.3 — Key Implementation Rules

1. **Never re-train models in app.py.** Only load from `models/`.
2. **All CSS** goes in `st.markdown("""<style>...</style>""", unsafe_allow_html=True)` at top.
3. **Use `st.session_state`** to persist results across reruns.
4. **Plotly charts** for probability visualization — not matplotlib (better interactivity).
5. **Confidence color logic:**
   ```python
   if confidence > 0.85:   color = high_certainty
   elif confidence > 0.65: color = moderate
   else:                   color = uncertain
   ```
6. **Error handling:** Wrap prediction in try/except. Show friendly error if models not found (tell user to run notebook first).
7. **Loading state:** Use `st.spinner("Analyzing article...")` during prediction.

### Step 6.4 — Streamlit Config

Create `.streamlit/config.toml`:
```toml
[theme]
base = "dark"
primaryColor = "#FF4B4B"
backgroundColor = "#0D1117"
secondaryBackgroundColor = "#161B22"
textColor = "#E6EDF3"
font = "sans serif"

[server]
headless = true
port = 8501
```

### Phase 6 Checkpoint ✓
- [ ] `streamlit run app.py` launches without error
- [ ] Model selector works
- [ ] Prediction returns correct result
- [ ] Probability bar chart renders
- [ ] Both tabs work
- [ ] App is mobile-responsive

---

---

# PHASE 7 — README.md

**File:** `README.md` (root level)

### Required Sections (in order):

```markdown
# FakeGuard — Fake News Detection System

## Overview
## Features
## Tech Stack (badges)
## Project Structure
## Dataset
## Installation
## Usage
  ### Running the Notebook
  ### Running the Streamlit App
## Model Performance
  (Insert actual metrics table after Phase 4)
## Text Preprocessing Pipeline
  (Diagram or step list)
## Screenshots
  (Reference: screenshots/app_demo.png — take one)
## Academic Context
## License
```

Use shields.io badges for tech stack:
```markdown
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.4.2-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.33.0-red)
```

---

---

# PHASE 8 — Final Polish & Submission Prep

### Step 8.1 — Notebook Finalization

The notebook is a primary submission artifact. Ensure:
- [ ] All cells have output (run top-to-bottom, `Kernel → Restart & Run All`)
- [ ] Every section has a markdown cell explanation (not just code)
- [ ] Introduction cell with: project title, your name, date, objective
- [ ] Conclusion cell: findings, best model, limitations, future work
- [ ] No dead/error cells anywhere
- [ ] Table of contents markdown at top (use links: `[1. Data Loading](#1-data-loading)`)

### Step 8.2 — Notebook Table of Contents Template

```markdown
## Table of Contents
1. [Setup & Imports](#1-setup--imports)
2. [Data Loading & Exploration](#2-data-loading--exploration)
3. [Text Preprocessing](#3-text-preprocessing)
4. [TF-IDF Vectorization](#4-tf-idf-vectorization)
5. [Model Training](#5-model-training)
6. [Model Evaluation](#6-model-evaluation)
7. [Model Comparison](#7-model-comparison)
8. [Inference & Testing](#8-inference--testing)
9. [Saving Artifacts](#9-saving-artifacts)
10. [Conclusion](#10-conclusion)
```

### Step 8.3 — Code Quality Pass

For each `.py` file in `src/`:
- [ ] Docstrings on every class and method
- [ ] Type hints on all function signatures
- [ ] No hardcoded paths — use `pathlib.Path`
- [ ] No `print()` in production code — use `logging`

### Step 8.4 — Submission Checklist

```
Final Submission Package:
├── ✅ notebooks/fake_news_detection.ipynb  (run, all outputs saved)
├── ✅ src/preprocessor.py
├── ✅ src/model_trainer.py
├── ✅ src/predictor.py
├── ✅ models/*.pkl                          (4 files)
├── ✅ data/processed/cleaned_data.csv
├── ✅ app.py
├── ✅ requirements.txt
├── ✅ README.md
└── ✅ PLAN.md
```

### Step 8.5 — Run Commands (Final Test)

```bash
# Test full pipeline
cd fake-news-detector
source venv/bin/activate

# Verify imports
python -c "from src.predictor import FakeNewsPredictor; p = FakeNewsPredictor(); print(p.predict('test text', 'Logistic Regression'))"

# Launch app
streamlit run app.py
```

---

---

# Appendix A — Common Errors & Fixes

| Error | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: nltk` | Not installed | `pip install nltk` |
| `LookupError: punkt` | NLTK data missing | `nltk.download('punkt')` |
| `FileNotFoundError: models/` | Notebook not run yet | Run notebook Phases 2–4 |
| `ValueError: not fitted` | Vectorizer loaded wrong | Re-save vectorizer after fit |
| `Streamlit: cache miss` | Model path wrong | Check relative vs absolute paths |

---

# Appendix B — Model Hyperparameter Reference

```python
# Best defaults for this task — do not change without reason
LogisticRegression(
    C=1.0,           # Regularization strength
    max_iter=1000,   # Ensure convergence
    solver='lbfgs',  # Default, works well with TF-IDF
    random_state=42
)

MultinomialNB(
    alpha=0.1        # Laplace smoothing — lower than default 1.0 for TF-IDF
)

RandomForestClassifier(
    n_estimators=100,   # 100 trees balances speed vs accuracy
    max_depth=None,     # Let trees grow fully
    random_state=42,
    n_jobs=-1           # Use all CPU cores
)

TfidfVectorizer(
    max_features=10000,    # Vocabulary cap
    ngram_range=(1, 2),    # Unigrams + bigrams
    min_df=2,              # Ignore extremely rare terms
    sublinear_tf=True      # Log normalization — helps with TF-IDF
)
```

---

*PLAN.md — FakeGuard Fake News Detection System*
*Build Phase: Sequential | Logic: Jupyter-first | UI: Streamlit*
