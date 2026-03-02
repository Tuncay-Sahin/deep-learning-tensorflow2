# Deep Learning Customer Behavior Prediction (Audiobooks Case Study)

## Overview

This project builds a deep learning model to predict whether a customer is likely to purchase audiobooks again based on behavioral features.

The goal is to transform raw customer interaction data into a decision-support system that can assist marketing targeting strategies.

Unlike tutorial-style notebooks, this repository includes a structured machine learning workflow and a production-ready prediction pipeline.

---
## Problem Context

Subscription-based digital platforms face a common challenge:

Not every customer who purchases once continues engaging with the product.

Understanding which customers are likely to return enables companies to:

- Improve marketing efficiency
- Reduce acquisition costs
- Personalize engagement strategies

In this project, a deep learning model is developed to predict whether a customer will purchase audiobooks again based on historical behavioral features.

The objective is not only model accuracy but also improving recall for repeat customers through decision threshold optimization aligned with business objectives.

---

## Business Problem

Audiobook platforms aim to identify customers who are more likely to make repeat purchases.

Accurate prediction enables:

* Better marketing targeting
* Reduced acquisition cost
* Improved customer retention strategies

---

## Approach

The project follows a complete machine learning lifecycle:

1. Data understanding and preparation
2. Feature preprocessing and scaling
3. Neural network training using TensorFlow
4. Validation and performance evaluation
5. Decision threshold optimization
6. Model artifact saving
7. Production-style inference pipeline

Key considerations:

* Class imbalance handling
* Precision–Recall tradeoff
* Threshold-based decision making

---

## Project Structure

```
deep-learning-tensorflow2/
│
├── data/
│   ├── processed/
│   │   └── standard_scaler.joblib
│   └── Audiobooks_data.csv
│
├── docs/
│
├── notebooks/
│   ├── 01_problem_understanding.ipynb
│   ├── 02_data_preprocessing.ipynb
│   └── 03_model_training_and_evaluation.ipynb
│
├── reports/
│   ├── audiobook_model.keras
│   └── best_threshold.pkl
│
├── src/
│   └── predict.py
│
├── environment.yml
├── README.md
└── .gitignore
```

---

## Model Pipeline

```
Raw Customer Data
        ↓
Feature Scaling (StandardScaler)
        ↓
Neural Network Model
        ↓
Probability Prediction
        ↓
Optimized Decision Threshold
        ↓
Customer Purchase Prediction
```

---

## Model Artifacts

The trained components are saved for reuse:

* Neural Network Model → `audiobook_model.keras`
* Feature Scaler → `standard_scaler.joblib`
* Optimized Threshold → `best_threshold.pkl`

This allows inference without retraining the model.

---

## Environment Setup

Create the environment using:

```
conda env create -f environment.yml
conda activate py3-TF2.0
```

---

## Quick Prediction Test

Run inference using the trained model:

```
python src/predict.py
```

Example output:

```
Prediction Result
----------------------------
Probability : 0.63
Threshold   : 0.58
Prediction  : 1
Customer likely to repeat
```

---

## Technologies Used

* Python
* TensorFlow / Keras
* NumPy
* Scikit-learn
* Joblib
* Jupyter Notebook

---

## Key Learning Outcomes

* Deep learning applied to tabular data
* Data preprocessing and scaling workflows
* Model evaluation beyond accuracy
* Threshold optimization for decision systems
* Transition from notebooks to inference pipeline

---

## Future Improvements

Potential extensions include:

* API deployment (FastAPI)
* Feature importance analysis dashboard
* Automated training pipeline
* Input validation for production inference

---

## Author

Tuncay Sahin
Data Science & Machine Learning Learning Journey
