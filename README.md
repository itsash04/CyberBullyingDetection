# 🧠 Cyberbullying Detection using NN & ML Techniques

## BERT | RoBERTa | LSTM | HANs | ML Classifiers with Hyperparameter Tuning

This repository presents a comprehensive **cyberbullying detection system** built on both **Neural Network architectures** and **Classical Machine Learning models**. The project aims to classify online social media content (tweets) as either **toxic** or **non-toxic**, using a carefully **balanced dataset**, **BERT-based tokenization**, and **hyperparameter-tuned models** for optimal performance.

---

## 🔍 Project Objective

* Detect cyberbullying in short-text formats (like tweets) using **text classification**.
* Compare the performance of **ML models** vs **Deep Learning architectures**.
* Balance dataset skew and improve detection robustness using **oversampling** and advanced **text preprocessing**.

---

## 🧠 Techniques Used

### 🧮 Machine Learning Models

All models were evaluated **before and after hyperparameter tuning**:

* ✅ **Logistic Regression**
* 🌲 **Random Forest Classifier**
* 💡 **Support Vector Machine (SVM)**
* 🚀 **XGBoost**
* 📈 **Gradient Boosting Classifier**

### 🧠 Neural Network Models

* 🧬 **BERT** (Fine-tuned, BERT Tokenizer)
* 📘 **RoBERTa** (Pretrained, Hugging Face Transformers)
* 🔁 **LSTM** (Long Short-Term Memory RNN)
* 🧩 **HAN** (Hierarchical Attention Networks)

---

## 📊 Data Preprocessing & Pipeline

### 📁 Dataset

* Source: **Twitter** — labeled for cyberbullying detection.
* Size: \~30,000 tweets
* Classes: `0 = Non-Toxic`, `1 = Cyberbullying`

### 🔧 Preprocessing Steps

* Removed:

  * Irrelevant symbols, URLs, mentions, hashtags
  * Non-alphanumeric characters
* Tokenization:

  * **BERT Tokenizer**
* Stopword Removal:

  * Used NLTK for English stopword filtering
* Data Balancing:

  * **Oversampling** with techniques like **SMOTE** to resolve class imbalance

---

## ⚙️ Technologies & Libraries

* **Language**: Python
* **ML**: `scikit-learn`, `xgboost`, `imbalanced-learn`
* **Deep Learning**: `TensorFlow`, `Keras`, `PyTorch`
* **NLP Models**: `transformers` (Hugging Face)
* **Preprocessing**: `nltk`, `re`, `pandas`, `numpy`

---

## 🧪 Evaluation Metrics

| Metric    | Description                                         |
| --------- | --------------------------------------------------- |
| Accuracy  | Correct predictions / total predictions             |
| Precision | True Positives / (True Positives + False Positives) |
| Recall    | True Positives / (True Positives + False Negatives) |
| F1-Score  | Harmonic mean of precision and recall               |

---

## 📊 Model Performance (Sample Summary)

| Model                     | Accuracy |
| ---------------------     | -------- |
| Logistic Regression       | 0.7980   |
| SVM (Tuned)               | 0.8033   |
| Random Forest (Tuned)     | 0.7867   | 
| Gradient Boosting (Tuned) | 0.7867   | 
| XGBoost (Tuned)           | 0.7958   | 
| LSTM                      | 0.8118   |
| BERT (Fine-tuned)         | 0.8417   | 
| RoBERTa (Fine-tuned)      | 0.8559   |
| HAN                       | 0.8037   |
---

## 🚀 Getting Started

### 🔧 Installation

```bash
git clone https://github.com/your-username/cyberbullying-nn-ml.git
cd cyberbullying-nn-ml
pip install -r requirements.txt
```

### 📦 Requirements

Key libraries used:

```txt
transformers
torch
tensorflow
scikit-learn
xgboost
imbalanced-learn
nltk
pandas
```

### ▶️ Run Example

```bash
# Train a tuned XGBoost model
python ml_models/tuned_xgboost.py

# Run BERT-based classifier
python nn_models/bert_classifier.py
```

Or open Jupyter notebooks for visual exploration.

---

## 📈 Visualizations

* Confusion matrices
* ROC Curves
* Loss/Accuracy plots for NN models
* Comparison charts of all models

All available under `results/`.

---

## 🎯 Future Enhancements

* Real-time toxicity detection on Twitter stream via Tweepy API
* Web app deployment using Flask/Streamlit
* Multilingual text classification
* Further experimentation with T5/DistilBERT or OpenAI models

---

## 🙌 Contributing

Pull requests, ideas, and discussions are welcome!
Please fork the repo and create a PR for any enhancements or bug fixes.

---

## ⭐ If you find this useful...

Give the project a ⭐ and share it with your peers. Let's make online spaces safer together!

---

