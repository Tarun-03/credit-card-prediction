import os
import sys
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from fastapi.staticfiles import StaticFiles

# -------------------------------
# Define request schema
# -------------------------------
class CreditData(BaseModel):
    gender: str
    age: int
    maritalStatus: str
    children: int
    familyMembers: int
    income: int
    incomeSource: str
    education: str
    employmentStatus: str
    yearsEmployed: int
    occupation: str
    ownCar: str
    ownRealty: str
    housingSituation: str
    creditHistoryLength: int = 0
    countLatePayments: int = 0
    percentageOnTimePayments: float = 100.0
    monthsSinceLastDelinquency: int = 999

# -------------------------------
# Initialize FastAPI
# -------------------------------
app = FastAPI(title="Credit Approval Prediction API")

origins = ["http://localhost:5173"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# Serve static files
# -------------------------------
# Create the 'static' directory if it doesn't exist
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)

# Mount the static directory to a URL path
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# -------------------------------
# Load CSV & Train Model
# -------------------------------
try:
    data_path = os.path.join(os.path.dirname(__file__), 'final_model_ready_data.csv')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"CSV file not found at {data_path}")
    data = pd.read_csv(data_path)

    # Identify target
    target_col = 'target' if 'target' in data.columns else 'RISK_ASSESSMENT'
    if target_col not in data.columns:
        raise ValueError("Target column not found in CSV")

    # Feature engineering
    if 'AGE_YEARS' not in data.columns:
        if 'DAYS_BIRTH' in data.columns:
            data['AGE_YEARS'] = abs(data['DAYS_BIRTH'] / 365)
        elif 'AGE' in data.columns:
            data['AGE_YEARS'] = data['AGE']

    if 'EMPLOYMENT_YEARS' not in data.columns:
        if 'DAYS_EMPLOYED' in data.columns:
            data['EMPLOYMENT_YEARS'] = abs(data['DAYS_EMPLOYED'] / 365)
        elif 'YEARS_EMPLOYED' in data.columns:
            data['EMPLOYMENT_YEARS'] = data['YEARS_EMPLOYED']

    # Ratios
    if 'AMT_INCOME_TOTAL' in data.columns and 'CNT_FAM_MEMBERS' in data.columns:
        denom = data['CNT_FAM_MEMBERS'].replace(0, np.nan)
        data['INCOME_PER_FAMILY'] = data['AMT_INCOME_TOTAL'] / denom

    if 'EMPLOYMENT_YEARS' in data.columns and 'AGE_YEARS' in data.columns:
        denom = data['AGE_YEARS'].replace(0, np.nan)
        data['EMPLOYMENT_RATIO'] = data['EMPLOYMENT_YEARS'] / denom

    if 'AMT_INCOME_TOTAL' in data.columns and 'EMPLOYMENT_YEARS' in data.columns:
        data['INCOME_PER_EMPLOYMENT_YEAR'] = data['AMT_INCOME_TOTAL'] / (data['EMPLOYMENT_YEARS'] + 1)

    # Encode categorical and boolean columns
    categorical_cols = data.select_dtypes(include='object').columns
    data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

    bool_cols = data.select_dtypes(include='bool').columns
    if len(bool_cols) > 0:
        data[bool_cols] = data[bool_cols].astype(int)

    # Features / target
    drop_cols = [c for c in ['ID', target_col] if c in data.columns]
    X = data.drop(drop_cols, axis=1)
    y = data[target_col].astype(int)

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))

    # Drop zero-variance columns
    constant_cols = X.columns[X.nunique() <= 1]
    X = X.drop(columns=constant_cols)

    # Train final model
    model = LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        class_weight='balanced',
        scale_pos_weight=10,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(X, y)
    model_columns = X.columns.tolist()

    # Compute optimal threshold based on training/validation split
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_curve
    X_train_val, X_val, y_train_val, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model.fit(X_train_val, y_train_val)
    y_val_prob = model.predict_proba(X_val)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_val, y_val_prob)
    youden = tpr - fpr
    opt_idx = np.argmax(youden)
    optimal_threshold = thresholds[opt_idx] if thresholds is not None and len(thresholds) > 0 else 0.5
    if np.isnan(optimal_threshold):
        optimal_threshold = 0.5

    print(f"Optimized decision threshold: {optimal_threshold:.4f}")

except Exception as e:
    print("Error loading/training model:", e)
    model = None
    model_columns = None
    optimal_threshold = 0.5

if model is not None and model_columns is not None:
    print(f"Model trained successfully with {len(model_columns)} features")
else:
    print("Model training failed. Check CSV path and preprocessing.")

# -------------------------------
# API endpoints
# -------------------------------
@app.get("/")
def home():
    return {"message": "Credit Card Approval Prediction API is running!"}

@app.post("/predict")
def predict_credit_approval(data: CreditData):
    if model is None or model_columns is None:
        return {"error": "ML model not loaded."}

    input_data = data.dict()

    # Feature engineering
    years_employed_raw = input_data['yearsEmployed']
    engineered_features = {
        'AGE_YEARS': input_data['age'],
        'EMPLOYMENT_YEARS': years_employed_raw,
        'INCOME_PER_FAMILY': input_data['income'] / input_data['familyMembers'] if input_data['familyMembers'] > 0 else 0,
        'INCOME_PER_EMPLOYMENT_YEAR': input_data['income'] / years_employed_raw if years_employed_raw > 0 else 0,
        'EMPLOYMENT_RATIO': years_employed_raw / input_data['age'] if input_data['age'] > 0 else 0,
    }

    processed_data = {**input_data, **engineered_features}
    processed_df = pd.DataFrame([processed_data])

    # Align columns
    final_df = pd.DataFrame(columns=model_columns)
    for col in model_columns:
        final_df[col] = processed_df[col] if col in processed_df.columns else 0

    # Predict
    prediction_proba = model.predict_proba(final_df)[:, 1]
    is_approved = bool(prediction_proba[0] < optimal_threshold)
    message = "Congratulations! Your application has been approved." if is_approved else "We regret to inform you that your application has been denied."

    return {
        "isApproved": is_approved,
        "message": message,
        "risk_probability_default": prediction_proba[0],
        "decision_threshold": optimal_threshold
    }

# -------------------------------
# Run server
# -------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
