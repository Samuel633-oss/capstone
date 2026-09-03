
from pathlib import Path
import json
import os
from threading import Lock

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
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "fraud_model_xgboost_tuned.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
CONFIG_PATH = MODEL_DIR / "model_config.json"
THRESHOLD_METRICS_PATH = MODEL_DIR / "threshold_metrics.json"

# Seed data ships with the app (small illustrative demo accounts — see
# scripts/seed_history_from_dataset.py for regenerating this from the
# REAL creditcard.csv dataset + the same account assignment used in
# training Cell 11, in an environment that has Kaggle access).
SEED_HISTORY_PATH = DATA_DIR / "account_history_seed.json"

# Live data accumulates here as real transactions come through the API,
# and is reloaded on every restart so history is no longer lost.
LIVE_HISTORY_PATH = DATA_DIR / "account_history_live.json"


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
# ACCOUNT HISTORY STORE (persistent, seeded)
# ============================================================
# Previously this was a plain in-memory dict: it started empty and
# reset on every restart, so "pull the account's transaction history"
# never actually had any history to pull unless you'd already submitted
# transactions earlier in the same process's lifetime.
#
# Now:
#   1. A small bundled seed file gives demo accounts (ACC-1001..1005)
#      real prior transactions out of the box.
#   2. Every transaction processed through the API is appended AND
#      written to disk (account_history_live.json), so history survives
#      restarts.
#   3. Seed and live data are kept in separate files and merged at load
#      time, so re-seeding never clobbers what the app has learned live.
#
# For real production use, swap this file-backed store for a proper
# database (Postgres, etc.) — this is a lightweight fix appropriate for
# a capstone/demo deployment, not a claim that it's production-grade.

_history_lock = Lock()


def _load_json(path: Path) -> dict:
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return {}


def _save_json(path: Path, data: dict):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _load_account_history_store() -> dict:
    store = _load_json(SEED_HISTORY_PATH)
    live = _load_json(LIVE_HISTORY_PATH)
    for account_id, txns in live.items():
        store.setdefault(account_id, [])
        store[account_id].extend(txns)
    return store


account_history_store = _load_account_history_store()


def add_to_history(account_id: str, txn: dict):
    with _history_lock:
        account_history_store.setdefault(account_id, [])
        account_history_store[account_id].append(txn)

        live_data = _load_json(LIVE_HISTORY_PATH)
        live_data.setdefault(account_id, [])
        live_data[account_id].append(txn)
        _save_json(LIVE_HISTORY_PATH, live_data)


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
# THRESHOLD METRICS (precision/recall/cost trade-off dashboard data)
# ============================================================

@app.get("/api/threshold-metrics")
def threshold_metrics():

    if not THRESHOLD_METRICS_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="threshold_metrics.json not found in model/"
        )

    with open(THRESHOLD_METRICS_PATH, "r") as f:
        return json.load(f)


# ============================================================
# ACCOUNT HISTORY (lets the UI show what the agent actually pulled)
# ============================================================

@app.get("/api/account-history/{account_id}")
def account_history(account_id: str):

    history = get_history(account_id)

    return {
        "account_id": account_id,
        "txn_count": len(history),
        "history": sorted(history, key=lambda h: h["Time"])
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
# INVESTIGATE (predict + pull history + draft report if flagged)
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

            # Pull the account's transaction history BEFORE adding this
            # transaction to it, so the agent is comparing the flagged
            # transaction against genuinely prior behavior.
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

            # Surface what the agent actually pulled, so it's visible in
            # the UI rather than only baked into the LLM's report text.
            result["history_used"] = {
                "txn_count": stats["txn_count"],
                "avg_amount": stats["avg_amount"],
                "max_amount": stats["max_amount"],
                "amount_ratio": stats["amount_ratio"],
                "recent_transactions": sorted(
                    history, key=lambda h: h["Time"]
                )[-5:]
            }

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


@app.get("/dashboard")
def dashboard():

    return FileResponse(
        STATIC_DIR / "dashboard.html"
    )
