# Credit Card Customer Churn Prediction

## Overview
An end-to-end, frontend-only AI/ML web application designed to predict whether a credit-card customer is likely to churn. Trained using Python/Keras on historical banking customer data and converted into TensorFlow.js format, the model runs **100% locally inside the browser** without any server-side prediction backend or API calls.

## Key Features
- **Frontend-Only Inference**: Operates directly in the browser via TensorFlow.js.
- **Strict Data Pipeline**: Reproduces training dataset preprocessing (one-hot encoding & StandardScaler) directly in JavaScript.
- **Interactive UI Dashboard**: Responsive form with 19 customer inputs organized into 4 intuitive sections, complete with validation, risk scoring, sample data loader, and form reset.

## Machine Learning Pipeline
1. Data Cleaning & Feature Selection (removal of non-predictive/target-leakage columns).
2. Categorical One-Hot Encoding (`pd.get_dummies(..., drop_first=True)` producing 32 input features).
3. `StandardScaler` fitted strictly on training data.
4. Artificial Neural Network (ANN) trained with Adam optimizer & binary cross-entropy loss.
5. In-browser prediction using model classification threshold of `0.30`.

## Quick Start (Frontend UI)
1. Clone this repository.
2. Open the project in VS Code.
3. Launch `frontend/index.html` using **VS Code Live Server** (or any static HTTP server).
4. Click **"Load Sample Customer"** and then **"Predict Churn"**.

---
*Project developed for ML / Deep Learning portfolio demonstration.*
