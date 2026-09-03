
// Preset sample V1-V28 values for demo purposes
// (real values pulled from actual dataset patterns)

const SAMPLES = {
  normal: {
    V1: -1.36, V2: -0.07, V3: 2.54, V4: 1.38, V5: -0.34,
    V6: 0.46, V7: 0.24, V8: 0.10, V9: 0.36, V10: 0.09,
    V11: -0.55, V12: -0.62, V13: -0.99, V14: -0.31, V15: 1.47,
    V16: -0.47, V17: 0.21, V18: 0.03, V19: 0.40, V20: 0.25,
    V21: -0.02, V22: 0.28, V23: -0.11, V24: 0.07, V25: 0.13,
    V26: -0.19, V27: 0.13, V28: -0.02,
    Amount: 149.62, Time: 0
  },
  fraud: {
    V1: -2.31, V2: 1.95, V3: -1.61, V4: 3.99, V5: -0.52,
    V6: -1.43, V7: -2.54, V8: 1.39, V9: -2.77, V10: -2.77,
    V11: 3.20, V12: -2.90, V13: -0.60, V14: -4.29, V15: 0.38,
    V16: -1.14, V17: -2.83, V18: -0.02, V19: 0.42, V20: 0.13,
    V21: 0.52, V22: 0.09, V23: 0.94, V24: -0.30, V25: -0.10,
    V26: -0.19, V27: 0.14, V28: 0.16,
    Amount: 1.00, Time: 95628
  }
};

let loadedFeatures = null;

const statusEl = document.getElementById("vfeatures-status");
const btnInvestigate = document.getElementById("investigate-btn");
const errorBox = document.getElementById("error-box");

document.querySelectorAll(".btn-sample").forEach(btn => {
  btn.addEventListener("click", () => {
    const key = btn.dataset.sample;
    loadedFeatures = { ...SAMPLES[key] };

    document.getElementById("time_val").value = loadedFeatures.Time;
    document.getElementById("amount_val").value = loadedFeatures.Amount;

    statusEl.textContent = `Loaded "${key}" sample — V1–V28 ready.`;
    statusEl.classList.add("ready");
    errorBox.classList.add("hidden");
  });
});

btnInvestigate.addEventListener("click", async () => {
  errorBox.classList.add("hidden");

  if (!loadedFeatures) {
    errorBox.textContent = "Load a sample first (V1–V28 values are required).";
    errorBox.classList.remove("hidden");
    return;
  }

  const accountId = document.getElementById("account_id").value.trim();
  const time = parseFloat(document.getElementById("time_val").value);
  const amount = parseFloat(document.getElementById("amount_val").value);

  if (!accountId) {
    errorBox.textContent = "Account ID is required.";
    errorBox.classList.remove("hidden");
    return;
  }

  const payload = {
    account_id: accountId,
    Time: time,
    Amount: amount,
    ...Object.fromEntries(
      Object.entries(loadedFeatures).filter(([k]) => k.startsWith("V"))
    )
  };

  btnInvestigate.disabled = true;
  btnInvestigate.textContent = "Investigating...";

  try {
    const res = await fetch("/api/investigate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Request failed");
    }

    const data = await res.json();
    renderResult(data);

  } catch (e) {
    errorBox.textContent = "Error: " + e.message;
    errorBox.classList.remove("hidden");
  } finally {
    btnInvestigate.disabled = false;
    btnInvestigate.textContent = "Investigate Transaction";
  }
});

function renderResult(data) {
  document.getElementById("result-empty").classList.add("hidden");
  document.getElementById("result-content").classList.remove("hidden");

  const probPct = Math.round(data.fraud_probability * 100);
  const probCircle = document.getElementById("prob-circle");
  const probValue = document.getElementById("prob-value");
  const flagStatus = document.getElementById("flag-status");
  const riskTier = document.getElementById("risk-tier");
  const reportSection = document.getElementById("report-section");
  const reportText = document.getElementById("report-text");

  probValue.textContent = probPct + "%";
  probCircle.className = "prob-circle";

  if (data.flagged) {
    flagStatus.textContent = "🚩 Flagged as Suspicious";
    flagStatus.style.color = "var(--danger)";
    probCircle.classList.add(data.risk_tier === "Critical" ? "critical" : "high");
  } else {
    flagStatus.textContent = "✅ Not Flagged";
    flagStatus.style.color = "var(--success)";
    probCircle.classList.add("low");
  }

  riskTier.textContent = data.risk_tier ? `Risk Tier: ${data.risk_tier}` : "";

  const historySection = document.getElementById("history-section");
  const historySummary = document.getElementById("history-summary");
  const historyTableBody = document.getElementById("history-table-body");

  if (data.history_used) {
    historySection.classList.remove("hidden");

    const h = data.history_used;
    if (h.txn_count === 0) {
      historySummary.textContent = "No prior transactions on file for this account — this is the first record the agent has for it.";
    } else {
      historySummary.textContent =
        `${h.txn_count} prior transaction(s) on file — avg $${h.avg_amount}, max $${h.max_amount}, ` +
        `this transaction is ${h.amount_ratio}× the account's average.`;
    }

    historyTableBody.innerHTML = "";
    (h.recent_transactions || []).forEach(txn => {
      const tr = document.createElement("tr");
      tr.innerHTML = `<td>${txn.Time}</td><td>$${txn.Amount.toFixed(2)}</td>`;
      historyTableBody.appendChild(tr);
    });
  } else {
    historySection.classList.add("hidden");
  }

  if (data.report) {
    reportSection.classList.remove("hidden");
    reportText.textContent = data.report;
  } else {
    reportSection.classList.add("hidden");
  }
}
