let metricsData = null;
let prChart = null;
let costChart = null;

const modelMeta = document.getElementById("model-meta");
const costFpInput = document.getElementById("cost-fp");
const costFnInput = document.getElementById("cost-fn");
const recommendationEl = document.getElementById("recommendation");
const tableBody = document.getElementById("metrics-table-body");

async function loadMetrics() {
  try {
    const res = await fetch("/api/threshold-metrics");
    if (!res.ok) throw new Error("Request failed: " + res.status);
    metricsData = await res.json();

    modelMeta.textContent =
      `${metricsData.model} — evaluated on ${metricsData.total_transactions.toLocaleString()} transactions ` +
      `(${metricsData.total_actual_fraud} actual frauds). Currently deployed threshold: ${metricsData.currently_deployed_threshold}.`;

    renderPRChart();
    recalculate();
  } catch (e) {
    modelMeta.textContent = "Could not load threshold metrics: " + e.message;
  }
}

function renderPRChart() {
  const labels = metricsData.thresholds.map(t => t.threshold);
  const precision = metricsData.thresholds.map(t => t.precision);
  const recall = metricsData.thresholds.map(t => t.recall);

  const ctx = document.getElementById("pr-chart").getContext("2d");
  prChart = new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: "Precision",
          data: precision,
          borderColor: "#5b8cff",
          backgroundColor: "#5b8cff",
          tension: 0.25,
        },
        {
          label: "Recall",
          data: recall,
          borderColor: "#4ade80",
          backgroundColor: "#4ade80",
          tension: 0.25,
        },
      ],
    },
    options: {
      responsive: true,
      scales: {
        y: { min: 0.6, max: 1.0, ticks: { color: "#9096a8" }, grid: { color: "#262a36" } },
        x: { title: { display: true, text: "Threshold", color: "#9096a8" }, ticks: { color: "#9096a8" }, grid: { color: "#262a36" } },
      },
      plugins: {
        legend: { labels: { color: "#e8e9ed" } },
      },
    },
  });
}

function computeCosts(costFp, costFn) {
  return metricsData.thresholds.map(t => ({
    ...t,
    total_cost: t.false_positives * costFp + t.false_negatives * costFn,
  }));
}

function renderCostChart(rows) {
  const labels = rows.map(r => r.threshold);
  const costs = rows.map(r => r.total_cost);

  if (costChart) {
    costChart.data.labels = labels;
    costChart.data.datasets[0].data = costs;
    costChart.update();
    return;
  }

  const ctx = document.getElementById("cost-chart").getContext("2d");
  costChart = new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: "Total Cost ($)",
          data: costs,
          borderColor: "#ffb84d",
          backgroundColor: "#ffb84d",
          tension: 0.25,
        },
      ],
    },
    options: {
      responsive: true,
      scales: {
        y: { ticks: { color: "#9096a8" }, grid: { color: "#262a36" } },
        x: { title: { display: true, text: "Threshold", color: "#9096a8" }, ticks: { color: "#9096a8" }, grid: { color: "#262a36" } },
      },
      plugins: {
        legend: { labels: { color: "#e8e9ed" } },
      },
    },
  });
}

function renderTable(rows) {
  const deployedThreshold = metricsData.currently_deployed_threshold;
  const minCost = Math.min(...rows.map(r => r.total_cost));

  tableBody.innerHTML = "";
  rows.forEach(r => {
    const tr = document.createElement("tr");
    if (r.threshold === deployedThreshold) tr.classList.add("deployed-row");
    if (r.total_cost === minCost) tr.classList.add("optimal-row");

    tr.innerHTML = `
      <td>${r.threshold}</td>
      <td>${(r.precision * 100).toFixed(1)}%</td>
      <td>${(r.recall * 100).toFixed(1)}%</td>
      <td>${r.flagged_count}</td>
      <td>${r.true_positives}</td>
      <td>${r.false_positives}</td>
      <td>${r.false_negatives}</td>
      <td>$${r.total_cost.toLocaleString()}</td>
    `;
    tableBody.appendChild(tr);
  });
}

function renderRecommendation(rows) {
  const deployedThreshold = metricsData.currently_deployed_threshold;
  const deployedRow = rows.find(r => r.threshold === deployedThreshold);
  const optimalRow = rows.reduce((a, b) => (b.total_cost < a.total_cost ? b : a));

  if (optimalRow.threshold === deployedThreshold) {
    recommendationEl.innerHTML =
      `At these costs, the currently deployed threshold ` +
      `<span class="threshold-figure">${deployedThreshold}</span> is already the cheapest option ` +
      `(≈$${deployedRow.total_cost.toLocaleString()} total cost on this validation set).`;
  } else {
    const savings = deployedRow.total_cost - optimalRow.total_cost;
    recommendationEl.innerHTML =
      `At these costs, threshold <span class="threshold-figure">${optimalRow.threshold}</span> minimizes total cost ` +
      `(≈$${optimalRow.total_cost.toLocaleString()} vs. ≈$${deployedRow.total_cost.toLocaleString()} at the currently ` +
      `deployed threshold ${deployedThreshold} — a difference of ≈$${savings.toLocaleString()} on this validation set).`;
  }
}

function recalculate() {
  if (!metricsData) return;
  const costFp = parseFloat(costFpInput.value) || 0;
  const costFn = parseFloat(costFnInput.value) || 0;

  const rows = computeCosts(costFp, costFn);
  renderCostChart(rows);
  renderTable(rows);
  renderRecommendation(rows);
}

costFpInput.addEventListener("input", recalculate);
costFnInput.addEventListener("input", recalculate);

loadMetrics();
