import json
import pickle
import random
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

# 1. Set Random Seeds for Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

def main():
    # Define paths using pathlib
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_PATH = BASE_DIR / "data" / "BankChurners.csv"
    MODEL_DIR = BASE_DIR / "model"
    MODEL_PATH = MODEL_DIR / "model.h5"
    SCALER_PATH = BASE_DIR / "scaler.pkl"
    PREPROCESSING_JSON_PATH = BASE_DIR / "preprocessing.json"

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset shape: {df.shape}")

    # 2. Target Mapping & Irrelevant Column Removal
    # Target: Attrition_Flag -> Existing Customer: 0, Attrited Customer: 1
    y = (df['Attrition_Flag'] == 'Attrited Customer').astype(int)

    drop_cols = [
        'CLIENTNUM',
        'Attrition_Flag',
        'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1',
        'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2'
    ]
    
    df_clean = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # 3. Identify Features
    num_cols = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio'
    ]

    cat_cols = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]

    # Store category unique values for frontend select elements
    category_values = {c: sorted(df[c].unique().tolist()) for c in cat_cols}

    # 4. Categorical One-Hot Encoding (drop_first=True)
    df_dummies = pd.get_dummies(df_clean[cat_cols], drop_first=True, dtype=float)

    # Combine Numerical Features FIRST, then One-Hot Encoded Categorical Features
    X = pd.concat([df_clean[num_cols], df_dummies], axis=1)
    feature_columns = list(X.columns)
    print(f"Total features created: {len(feature_columns)}")
    assert len(feature_columns) == 32, f"Expected 32 features, got {len(feature_columns)}"

    # 5. Train/Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    # 6. Fit StandardScaler ONLY on Training Data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 7. Model Architecture (32 -> Dense(16, relu) -> Dense(8, relu) -> Dense(1, sigmoid))
    model = Sequential([
        Input(shape=(32,), name="input_layer"),
        Dense(16, activation="relu", name="dense_1"),
        Dense(8, activation="relu", name="dense_2"),
        Dense(1, activation="sigmoid", name="output_layer")
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()

    # 8. Train the Model
    print("Training ANN model...")
    history = model.fit(
        X_train_scaled, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.1,
        verbose=1
    )

    # 9. Evaluate Model Performance
    y_pred_prob = model.predict(X_test_scaled).flatten()
    threshold = 0.30
    y_pred = (y_pred_prob >= threshold).astype(int)

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, pos_label=1))
    rec = float(recall_score(y_test, y_pred, pos_label=1))
    f1 = float(f1_score(y_test, y_pred, pos_label=1))
    cm = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(y_test, y_pred, output_dict=True)

    print("\n--- MODEL EVALUATION AT THRESHOLD 0.30 ---")
    print(f"Test Accuracy:  {acc:.4f}")
    print(f"Precision (Churn): {prec:.4f}")
    print(f"Recall (Churn):    {rec:.4f}")
    print(f"F1-Score (Churn):  {f1:.4f}")
    print("Confusion Matrix:")
    print(np.array(cm))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # 10. Save Artifacts
    # Keras Model
    model.save(MODEL_PATH)
    print(f"Saved model to: {MODEL_PATH}")

    # Scaler Pickle
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Saved scaler to: {SCALER_PATH}")

    # Export Preprocessing Metadata
    preprocessing_data = {
        "feature_columns": feature_columns,
        "numerical_columns": num_cols,
        "categorical_columns": cat_cols,
        "category_values": category_values,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "threshold": threshold,
        "evaluation_metrics": {
            "test_accuracy": acc,
            "precision_churn": prec,
            "recall_churn": rec,
            "f1_churn": f1,
            "confusion_matrix": cm,
            "classification_report": report
        }
    }

    with open(PREPROCESSING_JSON_PATH, "w") as f:
        json.dump(preprocessing_data, f, indent=4)
    print(f"Saved preprocessing metadata to: {PREPROCESSING_JSON_PATH}")

if __name__ == "__main__":
    main()
