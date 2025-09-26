# Natural Language Processing with Disaster Tweets (Kaggle)

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Kaggle](https://img.shields.io/badge/Kaggle-Competition-orange.svg)](https://www.kaggle.com/c/nlp-getting-started)

This is based on the [Kaggle competition: "Natural Language Processing with Disaster Tweets"](https://www.kaggle.com/c/nlp-getting-started).  
The task is to classify whether a tweet describes a real disaster (`label=1`) or not (`label=0`).

---

## 📋 Table of Contents
- [📌 Objective](#-objective)
- [📂 Dataset](#-dataset)
- [🛠️ Approaches](#️-approaches)
- [📊 Results](#-results)
- [📈 Error Analysis](#-error-analysis)
- [🚀 Future Work](#-future-work)
- [⚡ Installation](#-installation)
- [🚀 Usage](#-usage)

---

## 📌 Objective
Build model/s to automatically detect disaster-related tweets.  

The competition's evaluation metric is the **F1 score**:

\[
F1 = 2 \times \frac{precision \times recall}{precision + recall}
\]

- Precision = TP / (TP + FP)  
- Recall = TP / (TP + FN)

---

## 📂 Dataset
The competition provides three CSV files:
- `train.csv` → 7503 labeled tweets  
- `test.csv` → 3243 tweets for prediction  
- `sample_submission.csv` → template for submission  

The tweets are short, noisy, and often include hashtags, slang, abbreviations, sarcasm, or humor.

---

## 🛠️ Approaches

### 1. Multinomial Naive Bayes + TF-IDF
**Pipeline**  

**Preprocessing**
- Fix broken Unicode (`ftfy`)  
- Remove URLs  
- Decode HTML entities  
- Remove special characters, numbers, punctuation (strict regex, removes emojis)  
- Convert to lowercase  
- Normalize whitespace  
- Combine `text`, `keyword`, `location` into a `[combined_text]` column  

**TF-IDF Vectorization**
- Unigrams + bigrams (`ngram_range=(1,2)`)  
- Removed English stopwords  
- Vocabulary limited to top 10,000 features  

**Hyperparameter Tuning**
- Train/validation split: 80/20 stratified  
- Grid search over Laplace smoothing α values ([0.1, 0.3, 0.5, 1.0, 2.0])  
- Best α = 0.5  

---

### 2. Transformer Fine-Tuning (DistilRoBERTa-base)
A fine-tuned **DistilRoBERTa-base** (82M parameters), a distilled version of RoBERTa-base, for faster performance while retaining strong accuracy.

**Preprocessing**
- Fix broken Unicode (`ftfy`)  
- Replace links with `[URL]` token  
- Decode HTML entities  
- Normalize whitespace  
- Combine `text`, `keyword`, `location` into `[combined_text]`  

**Model Setup**
- Model: `distilroberta-base` (sequence classification with 2 labels)  
- Tokenizer: Hugging Face `AutoTokenizer`  
- Max length: 128 tokens  
- Trainer configuration:  
  - Weight decay = 0.01  
  - Epochs = 4  
  - Batch size = 16  
  - Optimizer = AdamW  
  - Evaluation metric = F1  

---

## 📊 Results

| Method                        | Kaggle F1 Score | Validation Accuracy |
|-------------------------------|-----------------|---------------------|
| Multinomial NB + TF-IDF (α=0.5) | **78.7%**       | 0.817 |
| DistilRoBERTa-base (fine-tuned) | **82.7%**       | 0.841 |

- **Naive Bayes**: Efficient and interpretable; useful feature importance (top positive/negative words).  
- **DistilRoBERTa**: Outperforms NB, generalizes better, achieves higher F1.  

---

## 📈 Error Analysis
- NB is lightweight and interpretable but struggles with nuanced tweets (sarcasm, humor).  
- DistilRoBERTa handles context better, reducing false negatives.  
- Trade-off: Transformers are computationally more expensive.  

---

## 🚀 Future Work
- Explore additional transformer models (BERT, RoBERTa, XLNet).  
- Hyperparameter optimization for transformer training.  
- Ensembling classical ML + transformers.  
- Deploy as a simple API or web demo.  
---

## ⚡ Installation

### Prerequisites
- Python 3.8+
- pip

### Setup
```bash
# Clone the repository
git clone https://github.com/serch3/nlp-disaster-tweets.git
cd nlp-disaster-tweets

# Create a virtual environment (optional but recommended)
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
# source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage

### Running the Notebooks
The project contains two Jupyter notebooks with different approaches:

1. **Approach 1: Multinomial Naive Bayes + TF-IDF**
   - Open `notebooks/approach_1.ipynb`
   - Run all cells to train and evaluate the model

2. **Approach 2: Transformer Fine-Tuning (DistilRoBERTa-base)**
   - Open `notebooks/approach_2.ipynb`
   - Run all cells to fine-tune and evaluate the transformer model

### Running the Python Scripts
Alternatively, you can run the models using the Python scripts in `src/`:

```bash
# Train Naive Bayes model
python src/main.py --model nb --train

# Train Transformer model
python src/main.py --model transformer --train

# Make predictions
python src/main.py --model nb --predict --output submission.csv
```


## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.