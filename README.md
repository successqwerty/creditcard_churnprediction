# Credit Card Customer Churn Prediction

## Overview
An end-to-end, frontend-only AI/ML web application designed to predict whether a credit-card customer is likely to churn. Trained using Python/Keras on historical banking customer data and converted into TensorFlow.js format, the model runs **100% locally inside the browser** without any server-side prediction backend or API calls.

## Problem Statement
Customer churn (attrition) is a critical challenge in the credit card industry. Identifying customers at high risk of churning enables financial institutions to proactively introduce targeted retention incentives.

## Dataset
Uses the `data/BankChurners.csv` dataset containing 10,127 customer records.

- **Target Column**: `Attrition_Flag`
  - `Existing Customer` -> `0` (No Churn)
  - `Attrited Customer` -> `1` (Churn)
- **Removed Columns**: `CLIENTNUM` (identifier), and two Naive Bayes target-leakage columns.

## Features (19 Raw Inputs)
1. **Customer Age**: Age in years (18–100)
2. **Gender**: Customer gender (M, F)
3. **Dependent Count**: Number of dependents
4. **Education Level**: Education status (College, Doctorate, Graduate, High School, Post-Graduate, Uneducated, Unknown)
5. **Marital Status**: Marital status (Divorced, Married, Single, Unknown)
6. **Income Category**: Annual income bracket (<$40K, $40K-$60K, $60K-$80K, $80K-$120K, $120K+, Unknown)
7. **Card Category**: Card tier (Blue, Silver, Gold, Platinum)
8. **Months on Book**: Period of relationship with bank in months
9. **Total Relationship Count**: Total number of products held
10. **Months Inactive in Last 12 Months**: Number of inactive months
11. **Contacts Count in Last 12 Months**: Number of contacts with bank
12. **Credit Limit**: Total credit card limit
13. **Total Revolving Balance**: Total revolving balance on credit card
14. **Average Open To Buy**: Open to buy credit line (Avg last 12 months)
15. **Amount Change Q4/Q1**: Change in transaction amount (Q4 vs Q1)
16. **Total Transaction Amount**: Total transaction amount (Last 12 months)
17. **Total Transaction Count**: Total transaction count (Last 12 months)
18. **Transaction Count Change Q4/Q1**: Change in transaction count (Q4 vs Q1)
19. **Average Utilization Ratio**: Average card utilization ratio (0 to 1)

## Data Preprocessing Pipeline
Raw Data (19 inputs)
  ↓
Target Mapping (`Existing Customer`=0, `Attrited Customer`=1)
  ↓
Categorical One-Hot Encoding (`pd.get_dummies(..., drop_first=True)`) -> 32 Features Total
  ↓
Train/Test Split (80% Train, 20% Test)
  ↓
StandardScaler (Fitted strictly on X_train)
  ↓
`preprocessing.json` Metadata Generation (Scaler mean, scale, feature names, threshold)

## Model Architecture
Artificial Neural Network (ANN) built with Keras/TensorFlow:
- **Input Layer**: 32 Features
- **Hidden Layer 1**: `Dense(16, activation="relu")`
- **Hidden Layer 2**: `Dense(8, activation="relu")`
- **Output Layer**: `Dense(1, activation="sigmoid")`
- **Optimizer**: Adam (`lr=0.001`)
- **Loss Function**: `binary_crossentropy`
- **Epochs**: 50 | **Batch Size**: 32
- **Classification Threshold**: `0.30`

## Evaluation Metrics (Actual Model Run Results)
Because churn is an imbalanced minority class (~16%), metrics prioritize **Recall** and **F1-Score** over accuracy alone.

| Metric | Score | Explanation |
| :--- | :--- | :--- |
| **Test Accuracy** | **91.56%** | Overall correct predictions |
| **Precision (Churn)** | **72.25%** | Percentage of predicted churners who actually churned |
| **Recall (Churn)** | **76.92%** | Percentage of actual churners detected by the model |
| **F1-Score (Churn)** | **74.52%** | Harmonic mean of Precision & Recall |

### Confusion Matrix
```
               Predicted No Churn    Predicted Churn
Actual No Churn        1605                96
Actual Churn            75                250
```

## Quick Start (Frontend UI)
1. Clone this repository.
2. Open the project in VS Code.
3. Launch `frontend/index.html` using **VS Code Live Server** (or any static HTTP server).
4. Click **"Load Sample Customer"** and then **"Predict Churn"**.
