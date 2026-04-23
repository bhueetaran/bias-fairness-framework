# ⚖️ Bias & Fairness Evaluation Framework in Machine Learning

End-to-end framework for detecting, evaluating, and mitigating bias in machine learning models, with integrated explainability using SHAP.

📊 Achieved ~97% reduction in demographic parity gap on real-world data with minimal accuracy trade-off.

---

## 🚀 Overview

This project implements a modular and reproducible pipeline for responsible AI. It enables:

* Detection of bias across sensitive groups
* Evaluation using standard fairness metrics
* Bias mitigation using constrained optimization
* Model interpretability using SHAP

The framework is dataset-agnostic and can be applied across different domains.

---

## ⚙️ Tech Stack

* Python
* Scikit-learn
* Fairlearn
* SHAP
* Pandas, NumPy
* Matplotlib, Seaborn

---

## 🧩 Pipeline

### 1. Bias Detection

* Data distribution analysis
* Group-wise outcome disparity
* Performance comparison across sensitive groups

### 2. Fairness Evaluation

* Demographic Parity
* Equalized Odds
* Group-wise accuracy

### 3. Bias Mitigation

* Exponentiated Gradient optimization
* Fairness–accuracy trade-off analysis

### 4. Explainability

* SHAP-based feature importance
* Group-wise explanation comparison
* Detection of proxy bias features

---

## 🧪 Methodology

### Stage 1: Synthetic Dataset (Validation)

A controlled dataset with intentionally introduced bias is used to validate the correctness of the framework under known conditions.

### Stage 2: Real Dataset (UCI Adult)

The same pipeline is applied to the UCI Adult Income dataset to demonstrate real-world applicability.

---

## 📊 Results

### Synthetic Dataset

| Metric             | Before | After |
| ------------------ | ------ | ----- |
| Accuracy           | 0.87   | 0.756 |
| Demographic Parity | 0.169  | 0.033 |
| Equalized Odds     | 0.338  | 0.353 |

✔️ Significant reduction in bias
✔️ Controlled accuracy trade-off

---

### Real Dataset (UCI Adult)

| Metric             | Before | After  |
| ------------------ | ------ | ------ |
| Accuracy           | 0.844  | 0.828  |
| Demographic Parity | 0.180  | 0.0055 |
| Equalized Odds     | 0.111  | 0.336  |

🔥 ~97% reduction in demographic parity gap
✔️ Minimal accuracy loss (~1.5%)
⚠️ Trade-off observed in equalized odds

---

## 🔍 Explainability with SHAP

* Identifies feature-level contributions to predictions
* Enables group-wise comparison to detect proxy bias
* Provides interpretability before and after mitigation

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
jupyter notebook bias_fairness_framework.ipynb
```

---

## 💡 Key Takeaways

* Demonstrates practical implementation of responsible AI
* Highlights fairness–accuracy trade-offs in ML systems
* Combines fairness metrics with explainability
* Applicable to both synthetic and real-world datasets
