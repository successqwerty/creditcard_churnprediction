/**
 * ChurnPredict — Frontend Inference Engine with TensorFlow.js
 * 100% Client-Side Machine Learning Prediction Pipeline
 */

// Global state variables
let tfjsModel = null;
let preprocessingMetadata = null;

// DOM Elements
const modelStatusBadge = document.getElementById('modelStatusBadge');
const statusText = document.getElementById('statusText');
const predictBtn = document.getElementById('predictBtn');
const sampleBtn = document.getElementById('sampleBtn');
const resetBtn = document.getElementById('resetBtn');
const churnForm = document.getElementById('churnForm');

const resultPlaceholder = document.getElementById('resultPlaceholder');
const resultContent = document.getElementById('resultContent');
const predictionStatusTitle = document.getElementById('predictionStatusTitle');
const riskBadge = document.getElementById('riskBadge');
const meterPercentage = document.getElementById('meterPercentage');
const meterFill = document.getElementById('meterFill');
const metricOutcome = document.getElementById('metricOutcome');
const metricRisk = document.getElementById('metricRisk');
const interpretationText = document.getElementById('interpretationText');

// Sample Customer Presets
const SAMPLES = {
    highRisk: {
        Customer_Age: 48,
        Gender: "F",
        Dependent_count: 2,
        Education_Level: "Uneducated",
        Marital_Status: "Single",
        Income_Category: "Less than $40K",
        Card_Category: "Blue",
        Months_on_book: 36,
        Total_Relationship_Count: 2,
        Months_Inactive_12_mon: 3,
        Contacts_Count_12_mon: 4,
        Total_Trans_Ct: 32,
        Total_Ct_Chng_Q4_Q1: 0.380,
        Credit_Limit: 2400,
        Total_Revolving_Bal: 0,
        Avg_Open_To_Buy: 2400,
        Total_Amt_Chng_Q4_Q1: 0.420,
        Total_Trans_Amt: 1450,
        Avg_Utilization_Ratio: 0.000
    },
    lowRisk: {
        Customer_Age: 44,
        Gender: "M",
        Dependent_count: 3,
        Education_Level: "Graduate",
        Marital_Status: "Married",
        Income_Category: "$80K - $120K",
        Card_Category: "Blue",
        Months_on_book: 36,
        Total_Relationship_Count: 5,
        Months_Inactive_12_mon: 1,
        Contacts_Count_12_mon: 2,
        Total_Trans_Ct: 82,
        Total_Ct_Chng_Q4_Q1: 0.820,
        Credit_Limit: 15400,
        Total_Revolving_Bal: 1850,
        Avg_Open_To_Buy: 13550,
        Total_Amt_Chng_Q4_Q1: 0.780,
        Total_Trans_Amt: 4850,
        Avg_Utilization_Ratio: 0.120
    }
};

/**
 * Initialize application by loading preprocessing metadata and TensorFlow.js model.
 */
async function initApp() {
    try {
        updateStatus("Loading preprocessing metadata...", "loading");

        // 1. Fetch preprocessing.json (try relative paths for Live Server compatibility)
        let jsonResponse;
        try {
            jsonResponse = await fetch('../preprocessing.json');
            if (!jsonResponse.ok) throw new Error();
        } catch {
            jsonResponse = await fetch('preprocessing.json');
        }

        if (!jsonResponse.ok) {
            throw new Error(`Failed to load preprocessing.json (HTTP ${jsonResponse.status})`);
        }
        preprocessingMetadata = await jsonResponse.json();
        console.log("Preprocessing metadata loaded successfully:", preprocessingMetadata);

        // 2. Load TensorFlow.js Model
        updateStatus("Loading TensorFlow.js model...", "loading");
        
        let modelPath = './tfjs_model/model.json';
        tfjsModel = await tf.loadLayersModel(modelPath);
        
        console.log("TensorFlow.js model loaded successfully!");
        console.log("Model Input Shape:", tfjsModel.inputs[0].shape);

        // 3. Mark ready in UI
        updateStatus("Model Ready", "ready");
        predictBtn.disabled = false;

    } catch (error) {
        console.error("Initialization error:", error);
        updateStatus("Unable to load AI model", "error");
        alert("Unable to load prediction model. Please run the project using a local HTTP server such as VS Code Live Server.");
    }
}

/**
 * Update model status dot & badge text
 */
function updateStatus(message, state) {
    statusText.innerText = message;
    const dot = modelStatusBadge.querySelector('.status-dot');
    dot.className = "status-dot";
    
    if (state === "loading") {
        dot.classList.add("pulsing");
    } else if (state === "ready") {
        dot.classList.add("ready");
    } else if (state === "error") {
        dot.classList.add("error");
    }
}

/**
 * Client-Side Input Validation
 */
function validateForm() {
    let isValid = true;

    const rules = {
        Customer_Age: { min: 18, max: 100 },
        Dependent_count: { min: 0 },
        Months_on_book: { min: 0 },
        Total_Relationship_Count: { min: 1 },
        Months_Inactive_12_mon: { min: 0, max: 12 },
        Contacts_Count_12_mon: { min: 0 },
        Total_Trans_Ct: { min: 0 },
        Total_Ct_Chng_Q4_Q1: { min: 0 },
        Credit_Limit: { min: 0 },
        Total_Revolving_Bal: { min: 0 },
        Avg_Open_To_Buy: { min: 0 },
        Total_Amt_Chng_Q4_Q1: { min: 0 },
        Total_Trans_Amt: { min: 0 },
        Avg_Utilization_Ratio: { min: 0, max: 1 }
    };

    // Clear previous errors
    document.querySelectorAll('.form-group').forEach(fg => fg.classList.remove('has-error'));
    document.querySelectorAll('.error-msg').forEach(em => em.style.display = 'none');

    // Validate each input field
    const inputs = churnForm.querySelectorAll('input, select');
    inputs.forEach(input => {
        const name = input.name;
        const valStr = input.value.trim();
        const parent = input.closest('.form-group');

        if (!valStr) {
            isValid = false;
            showFieldError(parent, "This field is required.");
            return;
        }

        if (rules[name]) {
            const numVal = parseFloat(valStr);
            const r = rules[name];
            if (isNaN(numVal)) {
                isValid = false;
                showFieldError(parent, "Invalid numeric value.");
            } else if (r.min !== undefined && numVal < r.min) {
                isValid = false;
                showFieldError(parent, `Value must be ≥ ${r.min}.`);
            } else if (r.max !== undefined && numVal > r.max) {
                isValid = false;
                showFieldError(parent, `Value must be ≤ ${r.max}.`);
            }
        }
    });

    return isValid;
}

function showFieldError(parentGroup, message) {
    parentGroup.classList.add('has-error');
    const errSpan = parentGroup.querySelector('.error-msg');
    if (errSpan) {
        errSpan.innerText = message;
        errSpan.style.display = 'block';
    }
}

/**
 * Extract raw inputs and construct 32-feature scaled array matching training pipeline
 */
function preprocessInputs() {
    // 1. Gather raw inputs from form
    const raw = {
        Customer_Age: parseFloat(document.getElementById('Customer_Age').value),
        Gender: document.getElementById('Gender').value,
        Dependent_count: parseFloat(document.getElementById('Dependent_count').value),
        Education_Level: document.getElementById('Education_Level').value,
        Marital_Status: document.getElementById('Marital_Status').value,
        Income_Category: document.getElementById('Income_Category').value,
        Card_Category: document.getElementById('Card_Category').value,
        Months_on_book: parseFloat(document.getElementById('Months_on_book').value),
        Total_Relationship_Count: parseFloat(document.getElementById('Total_Relationship_Count').value),
        Months_Inactive_12_mon: parseFloat(document.getElementById('Months_Inactive_12_mon').value),
        Contacts_Count_12_mon: parseFloat(document.getElementById('Contacts_Count_12_mon').value),
        Credit_Limit: parseFloat(document.getElementById('Credit_Limit').value),
        Total_Revolving_Bal: parseFloat(document.getElementById('Total_Revolving_Bal').value),
        Avg_Open_To_Buy: parseFloat(document.getElementById('Avg_Open_To_Buy').value),
        Total_Amt_Chng_Q4_Q1: parseFloat(document.getElementById('Total_Amt_Chng_Q4_Q1').value),
        Total_Trans_Amt: parseFloat(document.getElementById('Total_Trans_Amt').value),
        Total_Trans_Ct: parseFloat(document.getElementById('Total_Trans_Ct').value),
        Total_Ct_Chng_Q4_Q1: parseFloat(document.getElementById('Total_Ct_Chng_Q4_Q1').value),
        Avg_Utilization_Ratio: parseFloat(document.getElementById('Avg_Utilization_Ratio').value)
    };

    // 2. Build 32 feature vector according to preprocessing.json feature_columns order
    const featureCols = preprocessingMetadata.feature_columns;
    const rawVector = [];

    featureCols.forEach(col => {
        // Numerical columns
        if (col in raw) {
            rawVector.push(raw[col]);
        }
        // One-Hot Encoded Categorical columns (format: <ColName>_<CategoryValue>)
        else {
            // Find prefix match from categorical columns
            let matchedValue = 0;
            if (col.startsWith("Gender_")) {
                const val = col.replace("Gender_", "");
                if (raw.Gender === val) matchedValue = 1;
            } else if (col.startsWith("Education_Level_")) {
                const val = col.replace("Education_Level_", "");
                if (raw.Education_Level === val) matchedValue = 1;
            } else if (col.startsWith("Marital_Status_")) {
                const val = col.replace("Marital_Status_", "");
                if (raw.Marital_Status === val) matchedValue = 1;
            } else if (col.startsWith("Income_Category_")) {
                const val = col.replace("Income_Category_", "");
                if (raw.Income_Category === val) matchedValue = 1;
            } else if (col.startsWith("Card_Category_")) {
                const val = col.replace("Card_Category_", "");
                if (raw.Card_Category === val) matchedValue = 1;
            }
            rawVector.push(matchedValue);
        }
    });

    if (rawVector.length !== 32) {
        throw new Error(`Constructed feature vector length is ${rawVector.length}, expected 32.`);
    }

    // 3. Apply StandardScaler formula: (x - mean) / scale
    const mean = preprocessingMetadata.scaler_mean;
    const scale = preprocessingMetadata.scaler_scale;

    const scaledVector = rawVector.map((val, idx) => (val - mean[idx]) / scale[idx]);

    return scaledVector;
}

/**
 * Handle Predict Churn Form Submit
 */
churnForm.addEventListener('submit', async function(e) {
    e.preventDefault();

    if (!tfjsModel || !preprocessingMetadata) {
        alert("Model is not ready yet. Please wait.");
        return;
    }

    if (!validateForm()) {
        console.warn("Validation failed. Check form errors.");
        return;
    }

    try {
        // 1. Preprocess 32 features
        const scaledFeatures = preprocessInputs();

        // 2. Create TensorFlow.js Tensor (1 x 32)
        const inputTensor = tf.tensor2d([scaledFeatures], [1, 32]);

        // 3. Run In-Browser Prediction
        const predictionTensor = tfjsModel.predict(inputTensor);
        const probabilityArray = await predictionTensor.data();
        const churnProbability = probabilityArray[0];

        // 4. Dispose Tensors to prevent GPU memory leak
        inputTensor.dispose();
        predictionTensor.dispose();

        // 5. Render Prediction Results
        renderResult(churnProbability);

    } catch (err) {
        console.error("Prediction error:", err);
        alert("An error occurred during prediction: " + err.message);
    }
});

/**
 * Render Inference Result in Dashboard UI
 */
function renderResult(prob) {
    const threshold = preprocessingMetadata.threshold || 0.30;
    const percentageStr = (prob * 100).toFixed(1) + "%";

    resultPlaceholder.classList.add('hidden');
    resultContent.classList.remove('hidden');

    meterPercentage.innerText = percentageStr;
    meterFill.style.width = percentageStr;

    let isChurn = prob >= threshold;
    let riskLevel = "LOW";
    let riskClass = "low";
    let interpretation = "";

    if (prob < 0.30) {
        riskLevel = "LOW RISK";
        riskClass = "low";
        predictionStatusTitle.innerText = "Customer Unlikely to Churn";
        predictionStatusTitle.style.color = "var(--color-low)";
        metricOutcome.innerText = "No Churn";
        metricOutcome.style.color = "var(--color-low)";
        interpretation = `Based on the trained neural network model, this customer exhibits high engagement and low churn probability (${percentageStr}). They are currently predicted as unlikely to attrite.`;
    } else if (prob < 0.60) {
        riskLevel = "MEDIUM RISK";
        riskClass = "medium";
        predictionStatusTitle.innerText = "Customer Likely to Churn";
        predictionStatusTitle.style.color = "var(--color-medium)";
        metricOutcome.innerText = "Churn Detected";
        metricOutcome.style.color = "var(--color-medium)";
        interpretation = `The model estimates a moderate ${percentageStr} probability of churn (exceeding the ${threshold * 100}% decision threshold). Moderate retention incentives or follow-up communications are recommended.`;
    } else {
        riskLevel = "HIGH RISK";
        riskClass = "high";
        predictionStatusTitle.innerText = "Customer Highly Likely to Churn";
        predictionStatusTitle.style.color = "var(--color-high)";
        metricOutcome.innerText = "High Churn Risk";
        metricOutcome.style.color = "var(--color-high)";
        interpretation = `The model estimates a high ${percentageStr} probability of customer attrition. Immediate customer success intervention, targeted offers, or account review should be prioritized.`;
    }

    riskBadge.innerText = riskLevel;
    riskBadge.className = `risk-badge ${riskClass}`;
    metricRisk.innerText = riskLevel;
    interpretationText.innerText = interpretation;
}

/**
 * Load Sample Customer Data
 */
let currentSampleIndex = 0;
sampleBtn.addEventListener('click', function() {
    const sampleKey = currentSampleIndex % 2 === 0 ? 'highRisk' : 'lowRisk';
    currentSampleIndex++;

    const sample = SAMPLES[sampleKey];
    
    Object.keys(sample).forEach(key => {
        const elem = document.getElementById(key);
        if (elem) {
            elem.value = sample[key];
        }
    });

    // Clear validation errors
    document.querySelectorAll('.form-group').forEach(fg => fg.classList.remove('has-error'));
    document.querySelectorAll('.error-msg').forEach(em => em.style.display = 'none');

    console.log(`Loaded ${sampleKey} sample customer data.`);
});

/**
 * Reset Form Action
 */
resetBtn.addEventListener('click', function() {
    churnForm.reset();
    document.querySelectorAll('.form-group').forEach(fg => fg.classList.remove('has-error'));
    document.querySelectorAll('.error-msg').forEach(em => em.style.display = 'none');
    
    resultContent.classList.add('hidden');
    resultPlaceholder.classList.remove('hidden');
    meterFill.style.width = "0%";
    
    console.log("Form reset successfully.");
});

// Initialize app on DOM Content Loaded
document.addEventListener('DOMContentLoaded', initApp);
