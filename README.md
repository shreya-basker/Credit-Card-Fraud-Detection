# Credit Card Fraud Detection Using Machine Learning

A machine learning–based system to identify fraudulent credit card transactions using real-world transaction behavior and risk indicators.

## Problem Statement
Credit card fraud is a major challenge in digital payment systems due to the rarity of fraudulent transactions and the high cost of false positives. Detecting fraud requires identifying subtle patterns in transaction behavior while handling severe class imbalance.

## Dataset
A representative subset of a credit card transaction dataset was used for experimentation.  
The dataset includes realistic features such as transaction amount, transaction timing, device trust score, location mismatch, and transaction velocity.

Target variable:
- `is_fraud` (1 = Fraudulent transaction, 0 = Legitimate transaction)

## Approach
- Data cleaning and preprocessing
- Feature encoding and scaling
- Handling class imbalance
- Training baseline and ensemble machine learning models
- Evaluation using precision, recall, and F1-score

## Models Used
- Logistic Regression (baseline model)
- Random Forest Classifier (final model)

## Results
The Random Forest model demonstrated superior performance in detecting fraudulent transactions, particularly improving recall, which is critical for minimizing financial losses in fraud detection systems.

## Why This Project Matters
This project demonstrates how machine learning can be applied to real-world financial security problems. Such systems are directly applicable in banking, fintech platforms, and digital payment ecosystems to reduce fraud-related losses and improve transaction security.

## Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn

## Author
Shreya Basker
