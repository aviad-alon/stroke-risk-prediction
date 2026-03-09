# Stroke Risk Prediction

**Can we predict who is at risk of a stroke before it happens?**

Stroke is one of the leading causes of death and long-term disability worldwide, yet it often strikes without warning. This project uses a real clinical dataset to answer a practical question: given a patient's basic medical profile, can we reliably identify who is most at risk?

The core challenge is not simply achieving high accuracy - it is **catching as many real stroke cases as possible**, even at the cost of raising false alarms. In a clinical screening context, missing a high-risk patient is far worse than sending a healthy patient for an unnecessary follow-up. This principle shapes every modelling decision made in this project.

---

## Overview

The dataset contains **5,110 patient records** with features spanning demographics, lifestyle, and clinical indicators. The target variable is binary: did the patient suffer a stroke or not?

The project covers the full data mining pipeline:

1. **Exploratory Data Analysis** - understanding the data before touching it
2. **Preprocessing** - cleaning, imputing, encoding, and balancing
3. **Model Training** - Logistic Regression, Random Forest, XGBoost, Neural Network
4. **Evaluation** - recall-first metrics with a custom decision threshold
5. **Clustering** - discovering natural risk groups without labels

---

## Dataset

| Property | Value |
|---|---|
| Source | `healthcare-dataset-stroke-data.csv` |
| Records | 5,110 (5,109 after cleaning) |
| Features | 12 raw → 16 after encoding |
| Target | `stroke` (binary: 0 = no stroke, 1 = stroke) |
| Class balance | 4,860 negative (95.1%) / 249 positive (4.9%) |

### Features

| Feature | Type | Description |
|---|---|---|
| `age` | Continuous | Patient age in years |
| `avg_glucose_level` | Continuous | Average blood glucose |
| `bmi` | Continuous | 201 values imputed via KNN |
| `hypertension` | Binary | History of hypertension |
| `heart_disease` | Binary | History of heart disease |
| `ever_married` | Binary | Acts largely as a proxy for age |
| `gender` | Categorical | Male / Female |
| `work_type` | Categorical | Private, Self-employed, Govt, Never worked, Children |
| `Residence_type` | Categorical | Urban / Rural |
| `smoking_status` | Categorical | Never smoked, Formerly smoked, Smokes, Unknown |

> The dataset is heavily imbalanced (roughly 1:19). A naive classifier that always predicts "no stroke" would achieve 95% accuracy while being completely useless clinically.

---

## Methodology

### Preprocessing

- **Cleaning** - Removed the non-representative `Other` gender record and the uninformative `id` column. Applied domain logic to children: any patient under 10 with `smoking_status = Unknown` was corrected to `never smoked` (472 records).
- **Missing value imputation** - 201 missing BMI values were filled using a 10-nearest-neighbour imputer trained on age and glucose level.
- **Encoding** - Binary features mapped to 0/1; multi-category features one-hot encoded.
- **Train / test split** - 70/30 split.
- **Class imbalance** - SMOTE (Synthetic Minority Oversampling Technique) applied **to the training set only**, producing a balanced 1:1 split (3,888 vs 3,888). The test set remains at real-world distribution.
- **Scaling** - StandardScaler applied for Logistic Regression and the Neural Network. Tree-based models (Random Forest, XGBoost) use the unscaled data.

### Decision Threshold

All models are evaluated at a threshold of **0.20-0.25** rather than the default 0.5. This deliberately increases recall (fewer missed strokes) at the cost of lower precision (more false alarms). For a population-level screening tool, this trade-off is the correct one.

---

## Models and Results

| Model | Recall | ROC-AUC | Precision |
|---|---|---|---|
| Logistic Regression | 0.640 | 0.775 | 0.094 |
| Random Forest | 0.860 | 0.797 | 0.092 |
| XGBoost | 0.800 | 0.801 | 0.097 |
| Neural Network (ANN) | **0.900** | 0.791 | 0.081 |

> The Neural Network catches **90% of actual stroke cases** in the test set. The trade-off is that for every genuine stroke caught, roughly 11 healthy patients are also flagged - acceptable for a first-line screening tool that feeds into clinical review.

**Recommended models for this application:** Neural Network and Random Forest — both achieve recall ≥ 0.86 and ROC-AUC ≥ 0.79.

---

## Clustering

Unsupervised K-Means clustering (K = 7, selected via elbow method and silhouette analysis) was applied without using stroke labels, as an independent validation of the supervised findings.

| Cluster | Stroke Cases | Profile |
|---|---|---|
| 6 | 51 | Elderly, highest hypertension & heart disease - **maximum clinical risk** |
| 2 | 41 | Oldest demographic - age as an independent risk driver |
| 0 | 48 | Middle-aged with significant hypertension |
| 1 | 3 | Youngest, zero comorbidities - **healthy benchmark** |

The clustering naturally separated high-risk elderly patients from low-risk young ones without ever seeing the stroke labels, confirming that age, hypertension, and heart disease are the true drivers of risk.

---

## Key Findings

- **Age is the dominant predictor.** Stroke risk accelerates sharply after 60 regardless of other factors.
- **Hypertension and heart disease multiply risk significantly**.
- **Glucose level matters, BMI less so.** Glucose is a strong secondary predictor, BMI is most useful in combination with other features.
- **Clustering validates supervised results.** K-Means independently recovered the same risk stratification the labelled models learned, lending confidence to the feature importance findings.

> This project is a **screening tool**, not a diagnostic instrument. Its outputs should trigger clinical follow-up and never replace physician judgement.

---

## Limitations

- **Low precision** means substantial false positives. Acceptable for screening, but would require improvement before any real-world deployment.
- **Small dataset** - only 249 positive cases. Models would benefit significantly from additional real-world stroke records.
- **Static snapshot** - patient records represent a single point in time, longitudinal data would likely improve predictive power.

---

## Tech Stack

| Category | Libraries |
|---|---|
| Data manipulation | `pandas`, `numpy` |
| Visualisation | `matplotlib`, `seaborn` |
| Machine learning | `scikit-learn`, `xgboost` |
| Deep learning | `tensorflow` / `keras` |
| Imbalanced data | `imbalanced-learn` (SMOTE) |
| Statistics | `scipy` |

---

## Getting Started

### Prerequisites

- Python 3.8+
- Jupyter Notebook or JupyterLab

### Installation

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost tensorflow imbalanced-learn shap
```

### Running the notebook

```bash
jupyter notebook stroke-risk-prediction.ipynb
```

Run all cells in order. The notebook is self-contained and will download no external resources — just provide the `healthcare-dataset-stroke-data.csv` file in the same directory.

---

## Project Structure

```
.
├── stroke-risk-prediction.ipynb      # Full analysis notebook
└── healthcare-dataset-stroke-data.csv  # Patient dataset (required)
```
