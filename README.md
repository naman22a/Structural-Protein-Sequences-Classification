# 🧬 Structural Protein Sequences Classification

This project explores the application of **Machine Learning (ML)**, **Deep Learning (DL)**, and **Transfer Learning (TL)** techniques to classify **structural protein sequences**. By treating protein sequences like natural language, we leverage NLP-inspired approaches to identify and classify structural types based on sequence patterns.

---

## 📌 Project Overview

Proteins are composed of sequences of amino acids, and their structure plays a critical role in their function. This project applies **traditional ML algorithms**, **sequence-based neural networks**, and **state-of-the-art pretrained models** for classifying protein sequences into structural categories.

---

## 🧠 Model Architectures

### 🔹 Machine Learning Models

-   **Naive Bayes**
-   **XGBoost**
-   **Logistic Regression**
-   **K-Nearest Neighbors (KNN)**

### 🔹 Deep Learning Models

-   **Bidirectional LSTM (BiLSTM)**
-   **Convolutional Neural Network (CNN)**
-   **Recurrent Neural Network (RNN)**
-   **Gated Recurrent Unit (GRU)**

### 🔹 Transfer Learning Models

-   **ProtBERT**
-   **ESM2**

---

---

## 🔍 Explainability (XAI)

To better understand and interpret the predictions of the models, **Explainable AI (XAI)** techniques were applied using the **[LIME (Local Interpretable Model-Agnostic Explanations)](https://github.com/marcotcr/lime)** library.  
This helps in identifying which parts of the protein sequences contribute the most to the classification decision, providing biological interpretability alongside model performance.

## 📊 Evaluation Metrics

### 🔹 Machine Learning Results

| Model               | Accuracy | Precision | Recall | F1-Score |
| ------------------- | -------- | --------- | ------ | -------- |
| Naive Bayes         | 87.93%   | 88.17%    | 87.93% | 87.93%   |
| XGBoost             | 87.73%   | 87.71%    | 87.73% | 87.29%   |
| Logistic Regression | 89.97%   | 90.69%    | 89.97% | 90.04%   |
| KNN                 | 90.00%   | 91.23%    | 90.00% | 89.52%   |

---

### 🔹 Deep Learning Results

| Model  | Accuracy | Precision | Recall | F1-Score |
| ------ | -------- | --------- | ------ | -------- |
| BiLSTM | 88.09%   | 87.50%    | 88.09% | 87.54%   |
| CNN    | 69.07%   | 67.71%    | 69.07% | 67.16%   |
| RNN    | 39.39%   | 38.84%    | 39.39% | 36.46%   |
| GRU    | 71.57%   | 69.93%    | 71.57% | 69.67%   |

---

### 🔹 Transfer Learning Results

| Model    | Accuracy | Precision | Recall | F1-Score |
| -------- | -------- | --------- | ------ | -------- |
| ProtBERT | 61.02%   | 61.28%    | 61.02% | 59.15%   |
| ESM2     | 73.08%   | 74.03%    | 73.08% | 73.22%   |

---

## 🧪 Dataset

-   **Source**: [Kaggle - Protein Dataset](https://www.kaggle.com/datasets/shahir/protein-data-set)
-   **Classes**: 32
-   **Size**: 270,912 sequences

---

## 🛠️ Tech Stack

-   **Python**
-   **NumPy**
-   **Pandas**
-   **Scikit-learn**
-   **XGBoost**
-   **PyTorch**
-   **Transformers (Hugging Face)**
-   **LIME (for Explainable AI)**

---

## 🚀 How to Run (using Anaconda)

1. **Clone the repository**

```bash
git clone https://github.com/naman22a/Structural-Protein-Sequences-Classification
cd Structural-Protein-Sequences-Classification
```

2. **Create and activate a conda environment**

```bash
conda create -n protein-nlp python=3.10 -y
conda activate protein-nlp
```

3. **Install dependencies**

```bash
conda env create -f environment.yml
```

4. **Run the training script**

```bash
jupyter notebook
```

## 📫 Stay in touch

-   Author - [Naman Arora](https://namanarora.xyz)
-   Twitter - [@naman_22a](https://twitter.com/naman_22a)

## 🗒️ License

Structural Protein Sequences Classification is licensed under [GPL V3](./LICENSE)
