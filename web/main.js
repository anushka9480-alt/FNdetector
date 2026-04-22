const sampleArticle = {
  title: "Regional officials deny the viral subsidy memo circulating online",
  text:
    "Several social posts claimed the government had approved an emergency subsidy package overnight. " +
    "A spokesperson for the finance ministry said the memo being shared was fabricated and did not match " +
    "the formatting or approval process used for official releases. Independent reporters reviewing the " +
    "claim also found no record of the announcement on ministry websites or in parliamentary briefings.",
};

const API_BASE =
  window.__FN_DETECTOR_API_BASE__ || "https://Harman823-fndetector-backend.hf.space";
const HEALTH_URL = `${API_BASE}/health`;
const METRICS_URL = `${API_BASE}/metrics`;
const PREDICT_URL = `${API_BASE}/predict`;

let modelMetadataPromise = null;

const elements = {
  form: document.getElementById("prediction-form"),
  title: document.getElementById("title"),
  text: document.getElementById("text"),
  submitButton: document.getElementById("submit-button"),
  sampleButton: document.getElementById("sample-button"),
  clearButton: document.getElementById("clear-button"),
  healthBadge: document.getElementById("health-badge"),
  statusCopy: document.getElementById("status-copy"),
  metricGrid: document.getElementById("metric-grid"),
  verdictPill: document.getElementById("verdict-pill"),
  resultTitle: document.getElementById("result-title"),
  resultSummary: document.getElementById("result-summary"),
  resultModel: document.getElementById("result-model"),
  resultConfidence: document.getElementById("result-confidence"),
  resultLength: document.getElementById("result-length"),
  realScore: document.getElementById("real-score"),
  fakeScore: document.getElementById("fake-score"),
  realMeter: document.getElementById("real-meter"),
  fakeMeter: document.getElementById("fake-meter"),
};

function formatPercent(value) {
  return `${(value * 100).toFixed(1)}%`;
}

function normalizeText(value) {
  return String(value || "")
    .replace(/\u2019/g, "'")
    .replace(/\u201c/g, '"')
    .replace(/\u201d/g, '"')
    .replace(/\s+/g, " ")
    .trim();
}

function combineNewsText(title, text) {
  const normalizedTitle = normalizeText(title);
  const normalizedText = normalizeText(text);
  if (normalizedTitle && normalizedText) {
    return `${normalizedTitle}. ${normalizedText}`;
  }
  return normalizedTitle || normalizedText;
}

function normalizeLabel(label) {
  const normalized = String(label || "").toLowerCase();
  if (normalized.includes("fake") || normalized.endsWith("_1") || normalized === "label_1") {
    return "fake";
  }
  return "real";
}

function setHealthBadge(label, statusClass) {
  elements.healthBadge.textContent = label;
  elements.healthBadge.className = `status-badge ${statusClass}`;
}

function setStatusCopy(message) {
  if (elements.statusCopy) {
    elements.statusCopy.textContent = message;
  }
}

async function loadModelMetadata() {
  if (!modelMetadataPromise) {
    modelMetadataPromise = fetch(METRICS_URL).then(async (response) => {
      if (!response.ok) {
        throw new Error("Unable to load backend metrics.");
      }
      return response.json();
    });
  }

  return modelMetadataPromise;
}

function renderPrediction(prediction, metadata, combinedText) {
  const scoreMap = {
    real: Number(prediction?.scores?.real || 0),
    fake: Number(prediction?.scores?.fake || 0),
  };
  const predictedLabel = normalizeLabel(prediction?.predicted_label || prediction?.prediction);
  const confidence = Number(prediction?.confidence || Math.max(scoreMap.real, scoreMap.fake));
  const modelName =
    metadata?.snapshot?.model_name || metadata?.training_config?.model_name || "prajjwal1/bert-mini";

  elements.verdictPill.textContent = predictedLabel === "fake" ? "Likely fake" : "Likely real";
  elements.verdictPill.className = `verdict-pill ${predictedLabel}`;
  elements.resultTitle.textContent =
    predictedLabel === "fake"
      ? "Escalate this article for human review"
      : "Signals look closer to legitimate reporting";
  elements.resultSummary.textContent = `The model scored this article as ${predictedLabel} with ${formatPercent(
    confidence,
  )} confidence. Use this as a screening signal, not a final editorial decision.`;
  elements.resultModel.textContent = modelName;
  elements.resultConfidence.textContent = formatPercent(confidence);
  elements.resultLength.textContent = `${combinedText.length} chars`;
  elements.realScore.textContent = formatPercent(scoreMap.real);
  elements.fakeScore.textContent = formatPercent(scoreMap.fake);
  elements.realMeter.style.width = `${scoreMap.real * 100}%`;
  elements.fakeMeter.style.width = `${scoreMap.fake * 100}%`;
}

function renderMetrics(metadata) {
  const report = metadata.report || {};
  const testMetrics = report.test_metrics || {};
  const history = Array.isArray(report.history) ? report.history : [];
  const latestValidation = history[history.length - 1]?.validation || {};
  const cards = [
    {
      label: "Test accuracy",
      value: formatPercent(testMetrics.accuracy || 0),
      copy: "Overall correctness on the held-out test split served by the deployed backend model.",
      className: "accent-card",
    },
    {
      label: "Fake F1",
      value: formatPercent(testMetrics.f1_fake || 0),
      copy: "Balanced view of precision and recall for the fake-news class.",
      className: "pink-card",
    },
    {
      label: "Validation rows",
      value: String(report.validation_rows || 0),
      copy: "Validation examples used during checkpoint selection.",
      className: "yellow-card",
    },
    {
      label: "Train rows",
      value: String(report.train_rows || 0),
      copy: `Latest validation accuracy: ${formatPercent(latestValidation.accuracy || 0)}.`,
      className: "green-card",
    },
  ];

  elements.metricGrid.innerHTML = cards
    .map(
      (card) => `
        <article class="sticker-card metric-card ${card.className}">
          <span>${card.label}</span>
          <strong>${card.value}</strong>
          <p>${card.copy}</p>
        </article>
      `,
    )
    .join("");
}

async function loadDashboard() {
  try {
    setHealthBadge("Checking", "loading");
    const [healthResponse, metadata] = await Promise.all([
      fetch(HEALTH_URL).then(async (response) => {
        if (!response.ok) {
          throw new Error("Backend health check failed.");
        }
        return response.json();
      }),
      loadModelMetadata(),
    ]);

    renderMetrics(metadata);
    setHealthBadge("Ready", "ok");
    elements.resultModel.textContent = healthResponse.model || "Loaded";
    setStatusCopy(
      `Backend model ready. Active model ${healthResponse.model} with max length ${healthResponse.max_length}.`,
    );
  } catch (error) {
    setHealthBadge("Offline", "offline");
    setStatusCopy(
      "The Hugging Face backend could not be reached. Check that the Space has finished building and is publicly available.",
    );
  }
}

async function submitPrediction(event) {
  event.preventDefault();

  const combinedText = combineNewsText(elements.title.value, elements.text.value);
  if (!combinedText) {
    setStatusCopy("Paste an article body before running the classifier.");
    elements.text.focus();
    return;
  }

  elements.submitButton.disabled = true;
  elements.submitButton.textContent = "Analyzing...";
  setStatusCopy("Sending the article to the deployed detector.");

  try {
    const [predictionResponse, metadata] = await Promise.all([
      fetch(PREDICT_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          title: elements.title.value,
          text: elements.text.value,
        }),
      }).then(async (response) => {
        const payload = await response.json().catch(() => ({}));
        if (!response.ok) {
          throw new Error(payload.detail || "Prediction request failed.");
        }
        return payload;
      }),
      loadModelMetadata(),
    ]);

    renderPrediction(predictionResponse.prediction, metadata, combinedText);
    setStatusCopy("Prediction complete. Review the confidence split before making a decision.");
  } catch (error) {
    elements.verdictPill.textContent = "Request failed";
    elements.verdictPill.className = "verdict-pill fake";
    elements.resultTitle.textContent = "Prediction could not be completed";
    elements.resultSummary.textContent = error.message;
    setStatusCopy(
      "The remote backend could not complete the prediction. Confirm the Hugging Face Space is running and the model bundle loaded correctly.",
    );
  } finally {
    elements.submitButton.disabled = false;
    elements.submitButton.innerHTML = 'Run analysis <span class="button-icon" aria-hidden="true">→</span>';
  }
}

elements.sampleButton.addEventListener("click", () => {
  elements.title.value = sampleArticle.title;
  elements.text.value = sampleArticle.text;
  setStatusCopy("Sample article loaded. You can edit it before running analysis.");
});

elements.clearButton.addEventListener("click", () => {
  elements.form.reset();
  elements.verdictPill.textContent = "Awaiting input";
  elements.verdictPill.className = "verdict-pill neutral";
  elements.resultTitle.textContent = "Ready for review";
  elements.resultSummary.textContent =
    "Submit a sample to see confidence scores, the predicted class, and model details.";
  elements.resultModel.textContent = "Not loaded";
  elements.resultConfidence.textContent = "0%";
  elements.resultLength.textContent = "0 chars";
  elements.realScore.textContent = "0%";
  elements.fakeScore.textContent = "0%";
  elements.realMeter.style.width = "0%";
  elements.fakeMeter.style.width = "0%";
  setStatusCopy("Form cleared.");
});

elements.form.addEventListener("submit", submitPrediction);
loadDashboard();
