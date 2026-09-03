"""
Regenerate data/account_history_seed.json from the REAL creditcard.csv
dataset, using the exact same simulated account assignment as the
training notebook (Capstone_project.ipynb, Cell 11: np.random.seed(42)).

Why this exists:
-----------------
The Gotcha! app ships with a small illustrative demo seed file
(data/account_history_seed.json) covering 5 synthetic demo accounts
(ACC-1001..ACC-1005), because building this app didn't have access to
re-download the Kaggle dataset. Those demo transactions are NOT real
rows from the dataset — they're a placeholder so the "Investigate"
page shows populated history out of the box.

Run this script in an environment that DOES have Kaggle access (the
same Colab notebook used for training is the easiest option) to build
a seed file from the actual dataset instead, using the identical
account_id assignment your model's training used. Then copy the
output file into Gotcha/data/account_history_seed.json, replacing the
placeholder, before you deploy.

Usage:
    pip install kagglehub pandas numpy
    python seed_history_from_dataset.py

Output:
    account_history_seed.json — a JSON object mapping account_id (str)
    to a list of {"Time": ..., "Amount": ...} records for that
    account's LEGITIMATE (Class == 0) transactions. The flagged
    transaction under investigation is added to history separately at
    request time by main.py, so it's intentionally excluded here.
"""

import json
import os

import numpy as np
import pandas as pd
import kagglehub


def main():
    path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
    csv_path = os.path.join(path, "creditcard.csv")
    df = pd.read_csv(csv_path)

    # Must match Capstone_project.ipynb Cell 11 exactly, or account IDs
    # here won't line up with anything the trained model / notebook
    # analysis referred to.
    np.random.seed(42)
    n_accounts = 9500
    df["account_id"] = np.random.randint(0, n_accounts, size=len(df))

    legit = df[df["Class"] == 0]

    history = {}
    for account_id, group in legit.groupby("account_id"):
        history[str(account_id)] = (
            group[["Time", "Amount"]]
            .sort_values("Time")
            .to_dict("records")
        )

    out_path = "account_history_seed.json"
    with open(out_path, "w") as f:
        json.dump(history, f)

    print(f"Wrote history for {len(history)} accounts to {out_path}")
    print("Copy this file into Gotcha/data/account_history_seed.json to replace the demo placeholder.")
    print("NOTE: this file can be large (hundreds of thousands of records) — that's expected.")


if __name__ == "__main__":
    main()
