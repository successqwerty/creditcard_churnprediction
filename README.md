# Credit Card Customer Churn Prediction — ChurnPredict

An end-to-end, **frontend-only** AI/ML web application designed to predict whether a credit card customer is likely to churn (attrite). Trained using Python and Keras/TensorFlow on historical banking customer data and converted into TensorFlow.js format, the trained neural network model executes **100% locally inside the browser** without requiring any backend prediction server, API calls, node processes, or databases.

---

## 1. Project Overview & Business Motivation

Customer attrition (churn) is one of the most critical challenges in retail banking and credit card services. Acquiring a new credit card customer costs significantly more than retaining an existing one. By leveraging historical customer behavior, account attributes, and transaction metrics, financial institutions can predict high-risk churners and proactively offer retention incentives (e.g. fee waivers, upgraded rewards, increased credit limits).

This project delivers a **browser-native Machine Learning application** suitable for enterprise demonstration and Machine Learning / Deep Learning portfolio interviews.

---

## 2. Key Highlights & Technical Advantages

- **Zero Backend / Serverless Architecture**: Operates without FastAPI, Flask, Node.js, Express, SQL, or external APIs.
- **Client-Side TensorFlow.js Inference**: Model weights execute directly inside the user's browser runtime.
- **100% Privacy & Data Security**: Customer input data never leaves the client device.
- **Reproducible Machine Learning Pipeline**: Standardized scaling parameters (`scaler_mean` & `scaler_scale`) and feature mapping generated during Python training are exported to `preprocessing.json` and consumed identically by JavaScript.
- **Public & Multi-Device Accessibility**: Can be hosted as a static site (e.g., GitHub Pages) accessible by anyone on any device without requiring login credentials.

---

## 3. Dataset Overview (`BankChurners.csv`)

The model is trained on the dataset containing 10,127 credit card customer records.

- **Target Variable**: `Attrition_Flag`
  - `Existing Customer` → `0` (No Churn / Active)
  - `Attrited Customer` → `1` (Churn)
- **Removed Columns**:
  - `CLIENTNUM` (Unique customer ID — non-predictive)
  - `Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1` (Target leakage)
  - `Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2` (Target leakage)

---

## 4. Input Features (19 Customer Fields)

The application accepts all 19 raw customer attributes organized into 4 logical form sections:

### Section 1: Personal Information
1. **Customer Age**: Age in years (18 to 100)
2. **Gender**: Customer gender (`F`, `M`)
3. **Number of Dependents**: Count of financial dependents (0+)
4. **Education Level**: Highest education level (`Uneducated`, `High School`, `College`, `Graduate`, `Post-Graduate`, `Doctorate`, `Unknown`)
5. **Marital Status**: Marital status (`Single`, `Married`, `Divorced`, `Unknown`)
6. **Income Category**: Income bracket (`Less than $40K`, `$40K - $60K`, `$60K - $80K`, `$80K - $120K`, `$120K +`, `Unknown`)

### Section 2: Card & Account Information
7. **Card Category**: Credit card tier (`Blue`, `Silver`, `Gold`, `Platinum`)
8. **Months on Book**: Period of relationship with bank in months (0+)
9. **Total Relationship Count**: Total number of banking products held (1+)

### Section 3: Customer Activity (Last 12 Months)
10. **Months Inactive in Last 12 Months**: Inactive months in past year (0 to 12)
11. **Contacts Count in Last 12 Months**: Customer support contacts in past year (0+)
12. **Total Transaction Count**: Number of transactions in past 12 months (0+)
13. **Transaction Count Change (Q4/Q1)**: Ratio of Q4 transaction count to Q1 transaction count (0+)

### Section 4: Credit & Transaction Details
14. **Credit Limit**: Total credit line in USD (0+)
15. **Total Revolving Balance**: Revolving credit balance in USD (0+)
16. **Average Open To Buy**: Open-to-buy credit balance in USD (0+)
17. **Amount Change (Q4/Q1)**: Ratio of Q4 transaction amount to Q1 transaction amount (0+)
18. **Total Transaction Amount**: Total spend amount in past 12 months (0+)
19. **Average Utilization Ratio**: Average card credit utilization ratio (0.000 to 1.000)

---

## 5. Data Preprocessing & Feature Engineering

During Python training (`model/train.py`), the following pipeline transforms 19 raw inputs into **32 numerical features**:

1. **One-Hot Encoding**: Categorical features are encoded using `pd.get_dummies(..., drop_first=True)`:
   - `Gender`: `Gender_M` (Reference: `F`)
   - `Education_Level`: 6 dummy indicators (Reference: `College`)
   - `Marital_Status`: 3 dummy indicators (Reference: `Divorced`)
   - `Income_Category`: 5 dummy indicators (Reference: `$120K +`)
   - `Card_Category`: 3 dummy indicators (Reference: `Blue`)
2. **Feature Ordering**: Numerical features (14) are placed first, followed by categorical dummy features (18), forming exactly 32 features.
3. **StandardScaler Standardization**:
   \[
   x_{\text{scaled}} = \frac{x - \mu}{\sigma}
   \]
   `StandardScaler` is fitted **strictly on the 80% training split**. Mean (\(\mu\)) and scale (\(\sigma\)) values are saved into `preprocessing.json`.
4. **JavaScript Replication**: `frontend/script.js` loads `preprocessing.json` and performs the exact same one-hot encoding and Z-score scaling before passing tensors to TensorFlow.js.

---

## 6. Model Architecture & Training Configuration

The model is a 3-Layer Artificial Neural Network (ANN):

```
Input (32 Features)
    │
    ▼
Dense Layer 1 (16 Units, ReLU Activation)
    │
    ▼
Dense Layer 2 (8 Units, ReLU Activation)
    │
    ▼
Output Layer (1 Unit, Sigmoid Activation) -> Churn Probability [0.0, 1.0]
```

- **Framework**: Keras / TensorFlow 2.x
- **Optimizer**: Adam (`learning_rate = 0.001`)
- **Loss Function**: `binary_crossentropy`
- **Training Parameters**: Epochs = 50, Batch Size = 32, Train/Test Split = 80/20 (`random_state = 42`)
- **Classification Threshold**: `0.30`

---

## 7. Model Evaluation Metrics (Actual Training Results)

Because customer churn represents an imbalanced minority class (~16% of the dataset), model evaluation prioritizes **Recall** and **F1-Score** over accuracy alone. A lower decision threshold of `0.30` is selected to maximize churn detection.

| Metric | Score | Business Impact |
| :--- | :--- | :--- |
| **Test Accuracy** | **91.56%** | High overall correctness on unseen test data |
| **Precision (Churn)** | **72.25%** | 72.25% of customers flagged as churn risk actually churn |
| **Recall (Churn)** | **76.92%** | **76.92% of all actual churn customers are successfully identified** |
| **F1-Score (Churn)** | **74.52%** | Balanced metric accounting for both precision and recall |

### Confusion Matrix (Test Set: 2,026 Customers)
```
                          Predicted No Churn    Predicted Churn
Actual Existing Customer         1605                 96
Actual Attrited Customer           75                250
```

---

## 8. Frontend System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    User Browser Interface                   │
│  [HTML5 Form (19 Customer Inputs)] -> Client Validation     │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                    Frontend Preprocessing                   │
│  - One-Hot Categorical Encoding (32 Features)               │
│  - StandardScaler Formula: (x - mean) / scale               │
│  - Preprocessing parameters fetched from preprocessing.json  │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                 TensorFlow.js Engine (In-Browser)           │
│  - Model Loaded via tf.loadLayersModel('./tfjs_model/model.json')
│  - Tensor Creation: tf.tensor2d([scaled_array], [1, 32])    │
│  - In-Browser Forward Pass: model.predict(tensor)           │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                    Prediction UI Output                     │
│  - Churn Probability % (Gauge & Progress Fill)              │
│  - Threshold Check (>= 0.30 -> Churn)                       │
│  - Risk Badge: Low (<30%), Medium (30-60%), High (>=60%)    │
│  - Actionable Business Interpretation Insight               │
└─────────────────────────────────────────────────────────────┘
```

---

## 9. Project Directory Structure

```
creditcard_churnprediction/
│
├── data/
│   └── BankChurners.csv           # Historical banking dataset
│
├── model/
│   ├── train.py                   # Model training & evaluation script
│   └── model.h5                   # Trained Keras ANN model artifact
│
├── frontend/
│   ├── index.html                 # Modern dashboard UI
│   ├── style.css                  # Custom glassmorphic styling
│   ├── script.js                  # Preprocessing & TF.js inference script
│   └── tfjs_model/
│       ├── model.json             # Converted TF.js model architecture & manifest
│       └── group1-shard1of1.bin   # Binary model weight buffer
│
├── preprocessing.json             # Scaler params, feature columns & threshold
├── convert_model.py               # Keras to TensorFlow.js converter script
├── scaler.pkl                     # Python StandardScaler pickle
├── requirements.txt               # Dependencies for training & conversion
├── README.md                      # Project documentation
└── .gitignore                     # Git tracking exclusions
```

---

## 10. How To Run the Project

### Option A: Open Locally (Recommended for Testing & Live Server)
Because browser security blocks loading external `model.json` files via the `file://` protocol, the application must be served over HTTP. **No login, username, or password is required.**

1. Clone the repository:
   ```bash
   git clone https://github.com/successqwerty/creditcard_churnprediction.git
   cd creditcard_churnprediction
   ```
2. Open the project folder in **VS Code**.
3. Install the **Live Server** extension in VS Code.
4. Right-click [`frontend/index.html`](file:///c:/Users/hanum/OneDrive/Desktop/credircard_churnprediction/frontend/index.html) and select **"Open with Live Server"** (or run `python -m http.server 3000` from terminal and open `http://localhost:3000/frontend/index.html`).
5. Click **"Load Sample Customer"** and then **"Predict Churn"**.

### Option B: Deploy for Anyone on Any Device (GitHub Pages)
To make this project publicly accessible on mobile phones, tablets, or any computer worldwide:

1. Go to your GitHub repository settings: `https://github.com/successqwerty/creditcard_churnprediction/settings/pages`.
2. Under **Source**, select `main` branch and `/root` or `/frontend` folder.
3. Save. GitHub will provide a free public URL (e.g., `https://successqwerty.github.io/creditcard_churnprediction/frontend/`).
4. Anyone opening that link on any device can test the AI model instantly with **zero installation or setup**.

---

## 11. How To Retrain & Reconvert the Model

To retrain the Artificial Neural Network or modify hyper-parameters:

1. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Execute the training script:
   ```bash
   python model/train.py
   ```
3. Re-convert the trained Keras model into TensorFlow.js format:
   ```bash
   python convert_model.py
   ```

---

## 12. Interview Q&A Guide (Presentation Cheatsheet)

When presenting this project during a Machine Learning / Deep Learning technical interview:

- **Q: Why use an Artificial Neural Network (ANN) for tabular customer data?**
  *A: The dataset contains complex non-linear interactions between transaction count changes, revolving balances, and activity metrics. The 3-layer ANN with ReLU activations effectively learns non-linear decision boundaries.*
- **Q: Why fit StandardScaler only on X_train?**
  *A: Fitting scaler parameters on the entire dataset causes data leakage from test data into training data. Parameters (\(\mu, \sigma\)) must strictly be calculated from training data.*
- **Q: Why set the churn threshold to 0.30 instead of 0.50?**
  *A: In customer retention, the business cost of a False Negative (missing an actual churner) is much higher than a False Positive (offering a discount to a loyal customer). Lowering the threshold to 0.30 increases churn recall to 76.92%.*
- **Q: How does the model run without a backend server?**
  *A: The model topology and weight matrices are serialized into `model.json` and `.bin` buffers. TensorFlow.js loads these files into WebGL/CPU browser memory and executes matrix multiplication locally.*

---

## 13. Limitations & Future Roadmap

- **Feature Store Integration**: Currently preprocessed on the client side; future iterations could include automated schema validation.
- **Model Monitoring & Concept Drift**: As customer spending behavior changes over time, periodic retraining pipelines can be scheduled via GitHub Actions.
- **SMOTE Resampling**: Experimenting with Synthetic Minority Over-sampling Technique (SMOTE) during training could further boost churn precision.

---

## 14. License & Author
- **Repository**: [successqwerty/creditcard_churnprediction](https://github.com/successqwerty/creditcard_churnprediction)
- Developed for ML Portfolio & Demonstration.
