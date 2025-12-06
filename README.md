
# Conscientiousness Personality Trait Classification (Judging vs. Perceiving) using BERT-Large

This repository contains the complete implementation of a **BERT-Large based NLP system** for predicting the **Conscientiousness trait**—specifically the **Judging (J)** vs. **Perceiving (P)** dimension of the MBTI personality model—from user-generated text.  
In addition to the transformer model, the project evaluates multiple baselines using **TF-IDF**, **POS features**, **classical ML classifiers**, **LSTM/Bi-LSTM deep networks**, and extensive statistical validation across **three independent splits**.

---

##  Project Overview

This project aims to automatically classify text into **Judging (J)** or **Perceiving (P)** personality categories using:

- **BERT-Large (24-layer transformer)**  
- Weighted Cross-Entropy to handle class imbalance  
- 3 independent **70/30 train–test splits**  
- Full evaluation with **Accuracy, Precision, Recall, F1**  
- Statistical tests: **paired t-test**, **Wilcoxon**, **ANOVA**, **Friedman test**  
- Baseline comparisons:  
  - SVM, Naïve Bayes, Logistic Regression  
  - Decision Tree, KNN  
  - Random Forest, XGBoost, Gradient Boosting  
  - LSTM, Bi-LSTM (with embeddings)

---


##  Model Architecture

### BERT-Large (Transformer-Based Classifier)
- 24 Transformer layers  
- Sequence length: 512 tokens  
- Hidden size: 1024  
- Feed-forward size: 4096  
- 16 Attention heads  
- GELU activation  
- Dropout: 0.10  
- Optimizer: Adam  
- Loss: **Weighted Cross-Entropy**  
- Class weights: J:P ≈ **1.0 : 1.53**

---

##  Training Configuration

| Parameter | Value |
|----------|--------|
| Epochs | **100** |
| Learning Rate | **1e-5** |
| Train Batch Size | **16** |
| Eval Batch Size | **32** |
| Warmup Steps | **100** |
| Weight Decay | **0.01** |
| Evaluation Steps | **100** |
| Logging Steps | **100** |
| Early Stopping | **Patience = 10** |
| Mixed Precision | **FP16=True** |

---

##  Evaluation Protocol

The experiment uses:

✔ **Three independent 70/30 splits**  
✔ Validation created from 10% of training data  
✔ Metrics computed per split and aggregated:

- Mean  
- Standard deviation  
- 95% confidence intervals  

### Statistical Significance Tests
- Paired t-test  
- Wilcoxon signed-rank  
- One-way ANOVA  
- Friedman test  



---

##  Key Results

**BERT-Large (Proposed Model)**  
- **Accuracy:** 97.8% (±0.9)  
- **Precision:** 94.2%  
- **Recall:** 93.2%  
- **F1-Score:** 95.0%  

**Baselines**  
- TF-IDF + SVM: 80%  
- Ensemble (Gradient Boosting): 81%  
- LSTM w/ Sentence Embeddings: 89.6%  
- Bi-LSTM w/ Sentence Embeddings: **90.5%**

---

##  How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Training
```bash
python scripts/Conscientiousness_Personality_Trait_JP.py
```

### 3. Evaluate Model
```bash
python scripts/evaluate_model.py
```

---


##  Acknowledgements

This work uses:
- HuggingFace Transformers  
- PyTorch  
- Scikit-learn  
- Matplotlib & Seaborn  
- MBTI Dataset  
