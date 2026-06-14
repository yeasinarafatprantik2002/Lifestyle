const form = document.querySelector("#riskForm");
const resultPanel = document.querySelector("#resultPanel");

function riskClass(label) {
  const value = label.toLowerCase();
  if (value.includes("high")) return "risk-high";
  if (value.includes("moderate")) return "risk-moderate";
  return "risk-low";
}

function formToPayload(formElement) {
  const data = new FormData(formElement);
  return Object.fromEntries(data.entries());
}

function renderResult(result) {
  const recommendations = result.recommendations
    .map((item) => `<li>${item}</li>`)
    .join("");

  resultPanel.innerHTML = `
    <div>
      <span class="risk-badge ${riskClass(result.risk_label)}">${result.risk_label}</span>
      <div class="metric-grid">
        <div class="metric">
          <span>BMI</span>
          <strong>${result.bmi ?? "N/A"}</strong>
          <small>${result.bmi_category}</small>
        </div>
        <div class="metric">
          <span>Risk Score</span>
          <strong>${result.risk_score}</strong>
          <small>${result.model_prediction ? "Trained model used" : "Rule estimate"}</small>
        </div>
      </div>
      <h2>Recommendations</h2>
      <ul class="recommendations">${recommendations}</ul>
      <p class="note">${result.note}</p>
    </div>
  `;
}

function renderError(message) {
  resultPanel.innerHTML = `<div class="error">${message}</div>`;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const button = form.querySelector("button");
  button.disabled = true;
  button.textContent = "Analyzing...";

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(formToPayload(form)),
    });

    if (!response.ok) {
      throw new Error("Please check the form values and try again.");
    }

    renderResult(await response.json());
  } catch (error) {
    renderError(error.message || "Something went wrong.");
  } finally {
    button.disabled = false;
    button.textContent = "Analyze Risk";
  }
});
