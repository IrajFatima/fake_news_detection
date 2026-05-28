# 🚨 Fake News Detection System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg?logo=python\&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg?logo=jupyter\&logoColor=white)
![NLP](https://img.shields.io/badge/NLP-Preprocessing-green.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-red.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

### 🔍 A Complete Machine Learning Pipeline to Detect Fake News with **99.08% Accuracy**

**Logistic Regression** vs **Linear SVM** — Comparative Analysis

</div>

---

## 📖 Table of Contents

* [Project Overview](#-project-overview)
* [Key Results](#-key-results)
* [Technologies Used](#-technologies-used)
* [Dataset](#-dataset)
* [Project Structure](#-project-structure)
* [Environment Setup](#-environment-setup)
* [NLP Pipeline](#-nlp-pipeline)
* [Model Performance Visualizations](#-model-performance-visualizations)
* [Comparison: Logistic Regression vs SVM](#-comparison-logistic-regression-vs-svm)
* [Error Analysis](#-error-analysis)
* [Future Improvements](#-future-improvements)
* [Author](#-author)

---

## 🎯 Project Overview

This project builds a **complete Fake News Detection System** using Natural Language Processing (NLP) and Machine Learning. The system classifies news articles as **REAL** or **FAKE** by analyzing linguistic patterns, word usage, and textual features.

### Why Fake News Detection Matters?

> Misinformation spreads 6x faster than truth on social media. Automated detection systems are crucial for maintaining information integrity.

---

## 🏆 Key Results

| Model               | Accuracy   | Precision  | Recall     | F1 Score   | Test Errors |
| ------------------- | ---------- | ---------- | ---------- | ---------- | ----------- |
| **Linear SVM** ✅    | **99.08%** | **99.15%** | **99.08%** | **99.12%** | **83**      |
| Logistic Regression | 98.07%     | 98.67%     | 97.64%     | 98.15%     | 173         |

> 🎉 **Linear SVM outperformed Logistic Regression by 1.01% in accuracy and reduced errors by 90 misclassifications!**

---

## 🛠️ Technologies Used

| Category           | Technologies                                                                                                                   |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------ |
| **Language**       | ![Python](https://img.shields.io/badge/Python-3.8+-blue)                                                                       |
| **Data Handling**  | ![Pandas](https://img.shields.io/badge/Pandas-DataFrame-green) ![NumPy](https://img.shields.io/badge/NumPy-Arrays-blue)        |
| **Visualization**  | ![Matplotlib](https://img.shields.io/badge/Matplotlib-Plots-orange) ![Seaborn](https://img.shields.io/badge/Seaborn-Stats-red) |
| **NLP Processing** | ![NLTK](https://img.shields.io/badge/NLTK-Tokenization-purple) ![Regex](https://img.shields.io/badge/Regex-Cleaning-lightgrey) |
| **ML Models**      | ![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-TFIDF%7CLR%7CSVM-darkgreen)                                         |
| **Environment**    | ![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)                                                               |

---

## 📊 Dataset

**Source:** [ISOT Fake News Dataset](https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets)

### Dataset Statistics

```text
┌─────────────────────────────────────────┐
│                                         │
│   🟢 REAL News:  21,417 articles (47.7%)│
│   🔴 FAKE News:  23,481 articles (52.3%)│
│                                         │
│   Total:         44,898 articles        │
│   Train/Test:    80% / 20% split        │
│                                         │
└─────────────────────────────────────────┘
```

### 📁 Files

* `Fake.csv` - Misleading/fabricated news articles (Label: 1)
* `True.csv` - Verified legitimate news from Reuters (Label: 0)

### Download Dataset

```bash
pip install kaggle
kaggle datasets download -d emineyetm/fake-news-detection-datasets
tar -xf fake-news-detection-datasets.zip
```

---

## 📂 Project Structure

```text
fake_news_detection_system/
│
├── 📁 News_dataset/
│   ├── Fake.csv
│   └── True.csv
│
├── 📁 notebooks/
│   ├── fake_news_detection.ipynb
│   ├── fake_news_svm_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── 📁 venv/
│
└── README.md
```

---

## ⚙️ Environment Setup

### 1. Create Virtual Environment

```bash
python -m venv venv
```

### 2. Activate Virtual Environment

**Windows:**

```bash
venv\Scripts\activate
```

**Mac/Linux:**

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn nltk jupyter joblib
```

### 4. Create Jupyter Kernel

```bash
python -m ipykernel install --user --name=fakenewsenv
```

---

## 🔧 Windows Long Path Fix

If you encounter installation errors due to Windows path length limitations:

1. Open `gpedit.msc`
2. Navigate to: `Computer Configuration → Administrative Templates → System → Filesystem`
3. Enable **"Enable Win32 long paths"**
4. Restart your computer

---

## 🧠 NLP Pipeline

```text
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│   📰 Raw News Article                                               │
│          ↓                                                          │
│   🧹 Regex Cleaning → Remove Reuters bias, special characters       │
│          ↓                                                          │
│   📝 Tokenization → Split text into words (NLTK)                    │
│          ↓                                                          │
│   🚫 Stopword Removal → Remove common words (preserve negations)    │
│          ↓                                                          │
│   🌱 Stemming → Reduce words to root form (PorterStemmer)           │
│          ↓                                                          │
│   📊 TF-IDF Vectorization → Convert text to numerical features      │
│          ↓                                                          │
│   🤖 Machine Learning → Logistic Regression / Linear SVM            │
│          ↓                                                          │
│   ✅ Prediction → "REAL News" or "FAKE News"                        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 🔑 Custom Stopwords Strategy

Unlike standard NLP, this project **preserves** important words:

| Category           | Examples                    | Why Keep?                                  |
| ------------------ | --------------------------- | ------------------------------------------ |
| **Negations**      | `not`, `no`, `never`, `but` | Critical for detecting false claims        |
| **Pronouns**       | `they`, `we`, `you`         | Fake news uses specific pronoun densities  |
| **Interrogatives** | `why`, `how`, `which`       | Indicates questioning/speculative language |

---

## 📈 Model Performance Visualizations

### 📊 Dataset Distribution

```python
# Balanced dataset with 47.7% Real / 52.3% Fake news
```

### 🔥 Confusion Matrices

| Logistic Regression     | Linear SVM              |
| ----------------------- | ----------------------- |
| ✅ True Negatives: 4,222 | ✅ True Negatives: 4,244 |
| ❌ False Positives: 62   | ❌ False Positives: 40   |
| ❌ False Negatives: 111  | ❌ False Negatives: 43   |
| ✅ True Positives: 4,585 | ✅ True Positives: 4,653 |

### 📊 Performance Comparison Bar Chart

```text
        ┌────────────────────────────────────────────────────────┐
        │                                                        │
        │  1.00 ─┬─────────────────────────────────────────     │
        │        │  ████    ████    ████    ████                 │
        │  0.99 ─┤  ████    ████    ████    ████    🟢 SVM        │
        │        │  ████    ████    ████    ████    🔴 Logistic   │
        │  0.98 ─┤  ████    ████    ████    ████                 │
        │        │                                                │
        │  0.97 ─┴─────────────────────────────────────────     │
        │          Acc     Prec    Recall   F1                   │
        └────────────────────────────────────────────────────────┘
```

### 🎨 Performance Heatmap

| Metric    | Logistic Regression | Linear SVM |
| --------- | ------------------- | ---------- |
| Accuracy  | 0.9807              | **0.9908** |
| Precision | 0.9867              | **0.9915** |
| Recall    | 0.9764              | **0.9908** |
| F1 Score  | 0.9815              | **0.9912** |

### 📉 Error Analysis

```text
┌─────────────────────────────────────────────────────────┐
│                                                         │
│   Logistic Regression:  ████████████████░░  173 errors │
│   Linear SVM:           ████████░░░░░░░░░░   83 errors │
│                                                         │
│   🎯 Improvement: 90 FEWER ERRORS with SVM!            │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 🆚 Comparison: Logistic Regression vs Linear SVM

| Aspect                      | Logistic Regression | Linear SVM (Winner 🏆)  |
| --------------------------- | ------------------- | ----------------------- |
| **Approach**                | Probabilistic       | Margin Maximization     |
| **Decision Boundary**       | Soft                | Hard (with soft margin) |
| **Handles High Dimensions** | Good                | **Excellent**           |
| **Overfitting Risk**        | Moderate            | **Lower**               |
| **Training Speed**          | Fast                | Fast                    |
| **Test Accuracy**           | 98.07%              | **99.08%**              |

### ✅ Why SVM Performed Better:

> Text data processed with TF-IDF creates **linearly separable patterns**. SVM's **margin maximization** finds cleaner decision boundaries than Logistic Regression's probabilistic approach, especially for high-dimensional sparse vectors.

---

## 🚀 Future Improvements

| Priority  | Enhancement                        | Expected Gain      |
| --------- | ---------------------------------- | ------------------ |
| 🔴 High   | LSTM/GRU Deep Learning             | +1-2% accuracy     |
| 🔴 High   | BERT Transformer Fine-tuning       | +2-3% accuracy     |
| 🟡 Medium | Ensemble (RF + XGBoost + SVM)      | +0.5-1% accuracy   |
| 🟡 Medium | Hyperparameter Tuning (GridSearch) | +0.3-0.5% accuracy |
| 🟢 Low    | Streamlit Web App Deployment       | Accessibility      |
| 🟢 Low    | Real-time News API Integration     | Practical use      |

---

## 💻 Usage Example

```python
# Load the trained model and vectorizer
import joblib

svm_model = joblib.load("fake_news_svm_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

# Predict a news article
def predict_news(text):
    clean_text = preprocess_text(text)
    vector = tfidf.transform([clean_text])
    prediction = svm_model.predict(vector)[0]
    return "FAKE News" if prediction == 1 else "REAL News"

# Example
result = predict_news("Breaking: Government announces new policy changes...")
print(result)  # Output: REAL News or FAKE News
```

---

## 📚 Learning Objectives Covered

* [x] Regex-based text cleaning
* [x] NLP preprocessing (tokenization, stopwords, stemming)
* [x] Custom stopword strategy for fake news detection
* [x] TF-IDF vectorization (with n-grams)
* [x] Logistic Regression implementation
* [x] Linear SVM implementation
* [x] Train-test splitting with stratification
* [x] Model evaluation (accuracy, precision, recall, F1)
* [x] Confusion matrix visualization
* [x] Performance comparison charts
* [x] Model persistence with joblib
* [x] Prediction function development

---

## 👨‍💻 Author

<div align="center">

**Iraj Fatima**

Software Engineering Student | Web Development & AI Enthusiast

</div>

---

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

<div align="center">

### ⭐ If you found this project helpful, please give it a star! ⭐

</div>
