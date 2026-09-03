
from pathlib import Path
import json
import os

import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from pydantic import BaseModel

from groq import Groq


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "model"
STATIC_DIR = BASE_DIR / "static"

MODEL_PATH = MODEL_DIR / "fraud_model_xgboost_tuned.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
CONFIG_PATH = MODEL_DIR / "model_config.json"


# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load(
    MODEL_PATH
)

scaler = joblib.load(
    SCALER_PATH
)

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)


FEATURES = config["features"]

THRESHOLD = config.get(
    "threshold",
    0.7
)


# ============================================================
# GROQ SETUP
# ============================================================

groq_api_key = os.environ.get("GROQ_API_KEY")
client = Groq(api_key=groq_api_key) if groq_api_key else None
GROQ_MODEL = "openai/gpt-oss-120b"


# ============================================================
# IN-MEMORY ACCOUNT HISTORY STORE
# ============================================================
# NOTE: resets on app restart. Swap for a real database for production.

account_history_store = {}


def add_to_history(account_id: str, txn: dict):
    account_history_store.setdefault(account_id, [])
    account_history_store[account_id].append(txn)


def get_history(account_id: str):
    return account_history_store.get(account_id, [])


def summarize_history(history: list, current_amount: float):

    if not history:
        return {
            "txn_count": 0,
            "avg_amount": None,
            "max_amount": None,
            "amount_ratio": None
        }

    amounts = [h["Amount"] for h in history]
    avg_amount = sum(amounts) / len(amounts)
    max_amount = max(amounts)
    amount_ratio = current_amount / avg_amount if avg_amount > 0 else None

    return {
        "txn_count": len(history),
        "avg_amount": round(avg_amount, 2),
        "max_amount": round(max_amount, 2),
        "amount_ratio": round(amount_ratio, 2) if amount_ratio else None
    }


def get_risk_tier(probability: float):

    if probability >= 0.95:
        return "Critical", "Block immediately and escalate to senior fraud analyst"

    elif probability >= 0.85:
        return "High", "Hold for Review, lean toward blocking pending verification"

    else:
        return "Medium", "Hold for Review"


def draft_investigation_report(account_id, amount, time_val, probability, stats):

    if client is None:
        return "Groq API key not configured.", "Unknown", "N/A"

    risk_tier, mandated_action = get_risk_tier(probability)

    prompt = f"""Fraud analyst report for this flagged transaction. Be concise.

TRANSACTION: Account {account_id}, Amount ${amount:.2f}, Time {time_val}, Fraud probability {probability:.4f}
ACCOUNT HISTORY: {stats['txn_count']} prior txns, avg ${stats['avg_amount']}, max ${stats['max_amount']}, this txn is {stats['amount_ratio']}x average

MANDATORY POLICY: Risk tier = {risk_tier}. Action = {mandated_action}. State these exactly.

Write brief sections: 1) Case Summary 2) Behavioral Analysis 3) Risk Assessment ({risk_tier}) 4) Recommendation ({mandated_action} + short reasoning)."""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=350,
    )

    return response.choices[0].message.content, risk_tier, mandated_action


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Gotcha!",
    description="Transaction Fraud Classifier + AI Investigator Agent",
    version="1.0.0"
)


# ============================================================
# REQUEST MODEL
# ============================================================

class TransactionInput(BaseModel):

    account_id: str

    Time: float
    Amount: float

    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "service": "Gotcha!",
        "model": "fraud_model_xgboost_tuned"
    }


# ============================================================
# MODEL INFO
# ============================================================

@app.get("/api/model-info")
def model_info():

    return {
        "model": "Gotcha! Tuned XGBoost + SMOTE",
        "features": FEATURES,
        "feature_count": len(FEATURES),
        "fraud_threshold": THRESHOLD
    }


# ============================================================
# PREDICTION
# ============================================================

@app.post("/api/predict")
def predict(data: TransactionInput):

    try:

        values = data.model_dump()

        input_values = [
            values[feature]
            for feature in FEATURES
        ]

        X = pd.DataFrame(
            [input_values],
            columns=FEATURES
        )

        X[["Time", "Amount"]] = scaler.transform(
            X[["Time", "Amount"]]
        )

        probability = float(
            model.predict_proba(X)[0][1]
        )

        flagged = probability >= THRESHOLD

        txn_record = {
            "Time": values["Time"],
            "Amount": values["Amount"]
        }

        add_to_history(
            data.account_id,
            txn_record
        )

        return {

            "success": True,

            "account_id": data.account_id,

            "fraud_probability": round(
                probability,
                4
            ),

            "flagged": flagged,

            "threshold": THRESHOLD

        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ============================================================
# INVESTIGATE (predict + draft report if flagged)
# ============================================================

@app.post("/api/investigate")
def investigate(data: TransactionInput):

    try:

        values = data.model_dump()

        input_values = [
            values[feature]
            for feature in FEATURES
        ]

        X = pd.DataFrame(
            [input_values],
            columns=FEATURES
        )

        X[["Time", "Amount"]] = scaler.transform(
            X[["Time", "Amount"]]
        )

        probability = float(
            model.predict_proba(X)[0][1]
        )

        flagged = probability >= THRESHOLD

        result = {

            "success": True,

            "account_id": data.account_id,

            "fraud_probability": round(
                probability,
                4
            ),

            "flagged": flagged,

            "threshold": THRESHOLD

        }

        if flagged:

            history = get_history(data.account_id)

            stats = summarize_history(
                history,
                values["Amount"]
            )

            report, risk_tier, mandated_action = draft_investigation_report(
                data.account_id,
                values["Amount"],
                values["Time"],
                probability,
                stats
            )

            result["risk_tier"] = risk_tier
            result["mandated_action"] = mandated_action
            result["report"] = report

        txn_record = {
            "Time": values["Time"],
            "Amount": values["Amount"]
        }

        add_to_history(
            data.account_id,
            txn_record
        )

        return result

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ============================================================
# FRONTEND
# ============================================================

app.mount(
    "/static",
    StaticFiles(
        directory=STATIC_DIR
    ),
    name="static"
)


@app.get("/")
def home():

    return FileResponse(
        STATIC_DIR / "index.html"
    )
