"""
Credit Card Fraud Detection API
================================

Flask HTTP service that trains an XGBoost fraud-detection model on
startup (using the Kaggle "creditcardfraud" dataset via kagglehub) and
exposes it for real-time inference.

Endpoints:
    GET  /health   -> {"status": "ok"}
    POST /predict  -> {"fraud_probability": <float>, "is_fraud": <bool>}

This mirrors the pipeline built out in Capstone_project.ipynb:
    1. Load data from kagglehub (mlg-ulb/creditcardfraud)
    2. Stratified 70/15/15 train/val/test split
    3. StandardScaler fit on Time + Amount (V1-V28 are already PCA-scaled)
    4. SMOTE oversampling of the minority (fraud) class on the training set
    5. XGBoost classifier training
"""

import os
import logging

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fraud-api")

FEATURES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
SCALE_COLS = ["Time", "Amount"]
TARGET_COL = "Class"

# Populated by train_model() on startup.
model = None
scaler = None
model_ready = False


def load_dataset() -> pd.DataFrame:
    """Download (or reuse cached) creditcardfraud dataset via kagglehub."""
    import kagglehub

    path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
    csv_path = os.path.join(path, "creditcard.csv")
    df = pd.read_csv(csv_path)
    logger.info("Loaded dataset with shape %s", df.shape)
    return df


def train_model():
    """Train the fraud detection model and populate global model/scaler."""
    global model, scaler, model_ready

    logger.info("Starting model training...")

    df = load_dataset()

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    # 70% train, 15% val, 15% test (stratified on the target)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )

    logger.info("Train shape: %s | Val shape: %s | Test shape: %s",
                X_train.shape, X_val.shape, X_test.shape)

    # Scale Time and Amount; V1-V28 are already PCA-scaled.
    local_scaler = StandardScaler()

    X_train_scaled = X_train.copy()
    X_val_scaled = X_val.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[SCALE_COLS] = local_scaler.fit_transform(X_train[SCALE_COLS])
    X_val_scaled[SCALE_COLS] = local_scaler.transform(X_val[SCALE_COLS])
    X_test_scaled[SCALE_COLS] = local_scaler.transform(X_test[SCALE_COLS])

    # Balance the training set only.
    smote = SMOTE(sampling_strategy=1.0, random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(
        X_train_scaled, y_train
    )

    logger.info("Resampled train shape: %s", X_train_resampled.shape)

    clf = XGBClassifier(eval_metric="logloss", random_state=42, n_jobs=-1)
    clf.fit(X_train_resampled, y_train_resampled)

    y_pred_val = clf.predict(X_val_scaled)
    logger.info(
        "Validation performance:\n%s",
        classification_report(y_val, y_pred_val, target_names=["Legit", "Fraud"]),
    )

    model = clf
    scaler = local_scaler
    model_ready = True

    logger.info("Model training complete.")


app = Flask(__name__)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    if not model_ready:
        return jsonify({"error": "model is not ready yet"}), 503

    payload = request.get_json(silent=True)
    if not payload:
        return jsonify({"error": "request body must be JSON"}), 400

    missing = [f for f in FEATURES if f not in payload]
    if missing:
        return jsonify({"error": f"missing features: {missing}"}), 400

    try:
        input_values = [float(payload[f]) for f in FEATURES]
    except (TypeError, ValueError) as exc:
        return jsonify({"error": f"invalid feature value: {exc}"}), 400

    X = pd.DataFrame([input_values], columns=FEATURES)
    X[SCALE_COLS] = scaler.transform(X[SCALE_COLS])

    probability = float(model.predict_proba(X)[0][1])

    return jsonify({
        "fraud_probability": round(probability, 4),
        "is_fraud": probability >= 0.5,
    })


if __name__ == "__main__":
    train_model()

    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
