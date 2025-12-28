# Bias & Fairness Evaluation Framework

This repository presents a modular and reproducible framework for detecting,
evaluating, and mitigating bias in machine learning models, with integrated
explainability using SHAP.

The framework follows responsible AI principles and is designed to be
dataset-agnostic, enabling its application across diverse domains.

---

## 🧩 Framework Components

The system consists of the following stages:

- **Bias Detection**
  - Data distribution analysis
  - Group-wise outcome disparity
  - Performance comparison across sensitive groups

- **Fairness Evaluation**
  - Demographic Parity
  - Equalized Odds
  - Group-wise accuracy analysis

- **Bias Mitigation**
  - Fairness-constrained optimization using the Exponentiated Gradient method
  - Explicit analysis of fairness–accuracy trade-offs

- **Explainability**
  - SHAP-based global feature importance
  - Group-wise SHAP comparison to identify proxy bias
  - Post-mitigation interpretability analysis

---

## 🧪 Methodology

The framework is evaluated in two stages to ensure methodological rigor:

### Stage 1: Synthetic Dataset (Framework Validation)
A controlled synthetic dataset with intentionally introduced bias is used to
validate the correctness and robustness of the framework. This enables
reproducible testing under known bias conditions.

### Stage 2: Real Dataset (UCI Adult Benchmark)
The same pipeline is applied to the UCI Adult Income dataset, a standard
benchmark in fairness research, to demonstrate applicability to real-world
bias scenarios involving sensitive attributes.

---

## ⚖️ Fairness Metrics

The following metrics are used for evaluation:

- Demographic Parity Difference
- Equalized Odds Difference
- Group-wise accuracy comparison

Both pre- and post-mitigation results are analyzed to quantify fairness
improvements.

---

## 🔍 Explainability with SHAP

SHAP is employed to:
- Interpret model decisions at the feature level
- Compare feature influence across sensitive groups
- Diagnose proxy features contributing to biased outcomes

For fairness-constrained models, SHAP explanations are generated using a
representative predictor from the learned ensemble.

---

## 🔁 How to Use

1. Clone the repository and install dependencies using `requirements.txt`.
2. Run the notebook `bias_fairness_framework.ipynb` to execute the full bias and fairness pipeline.

### Adapting to New Data
To use a custom dataset, replace the data-loading cell in **Stage 2** and define:
- feature matrix (`X`)
- target variable (`y`)
- sensitive attribute

All fairness metrics, mitigation, and SHAP explanations will be computed automatically.
