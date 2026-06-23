# Lifestyle Health Risk Predictor

A FastAPI web application for lifestyle-based health risk prediction and personalized recommendations. The website collects user lifestyle and health information, calculates BMI, predicts risk using a trained machine-learning model, and shows practical recommendations on the same page.

## Features

- One-page HTML, CSS, and JavaScript frontend
- FastAPI backend
- Automatic model training when the server starts
- Excel dataset support
- BMI calculation
- Risk prediction: `Low Risk`, `Moderate Risk`, or `High Risk`
- Personalized lifestyle recommendations
- Model comparison report saved as CSV
- Evaluation plots for report figures
- Trained model saved with `joblib`

## Project Structure

```text
Lifestyle/
|-- data/
|   |-- Data.xlsx
|   |-- lifestyle_health_risk_labeled_dataset.csv
|   `-- model_results.csv
|-- static/
|   |-- app.js
|   `-- styles.css
|-- templates/
|   `-- index.html
|-- artifacts/
|   `-- lifestyle_risk_model.joblib
|-- report_figures/
|   `-- evaluation/
|-- life_style_train.py
|-- main.py
|-- evaluation_plots.py
|-- requirements.txt
|-- requirements-dev.txt
|-- vercel.json
|-- .python-version
|-- .gitignore
`-- README.md
```

## Requirements

- Python 3.12 is used for Vercel production through `.python-version`.
- Python 3.11 or newer is recommended for local development.
- Python 3.14 may work locally, but some optional ML libraries can have compatibility issues depending on package availability.

## Setup

Install runtime dependencies:

```bash
pip install -r requirements.txt
```

Install optional development/evaluation dependencies:

```bash
pip install -r requirements-dev.txt
```

`requirements.txt` is intentionally kept production-focused for Vercel. Plotting and optional XGBoost support live in `requirements-dev.txt`.

## Dataset

Put the final training dataset here:

```text
data/Data.xlsx
```

The training logic expects the Excel file to contain 19 columns in this order:

1. Timestamp
2. Age
3. Gender
4. Height in cm
5. Weight in kg
6. Fruits and vegetables frequency
7. Fast food frequency
8. Water intake
9. Sugary drinks frequency
10. Exercise frequency
11. Exercise type
12. Sleep hours
13. Stress level
14. Smoking
15. Alcohol
16. Chronic conditions
17. Family history of heart disease
18. Fasting blood glucose
19. Overall health

## Run the Server

Use this command:

```bash
python -m uvicorn main:app --reload
```

Open the website:

```text
http://127.0.0.1:8000
```

When the server starts locally, it automatically trains the model using:

```text
data/Data.xlsx
```

After training, it saves:

```text
artifacts/lifestyle_risk_model.joblib
data/lifestyle_health_risk_labeled_dataset.csv
data/model_results.csv
```

In Vercel production, the app does not train during cold start. It loads the committed pre-trained model from:

```text
artifacts/lifestyle_risk_model.joblib
```

## Manual Training

You can also train without starting the website:

```bash
python life_style_train.py
```

This will clean the dataset, calculate BMI, create risk labels, train multiple models, save the best model, and write result files.

Before deploying to Vercel, run manual training once and commit the updated model artifact:

```text
artifacts/lifestyle_risk_model.joblib
```

## Evaluation Figures

Generate model evaluation figures and metrics for the project report:

```bash
python evaluation_plots.py
```

The script evaluates:

- KNN
- Decision Tree
- Random Forest
- Logistic Regression
- SVM
- ANN / MLPClassifier
- XGBoost, if installed
- Hybrid ANN + Random Forest

Outputs are saved in:

```text
report_figures/evaluation/
```

Generated files include:

```text
confusion_matrix_KNN.png
confusion_matrix_Decision_Tree.png
confusion_matrix_Random_Forest.png
confusion_matrix_Logistic_Regression.png
confusion_matrix_SVM.png
confusion_matrix_ANN.png
confusion_matrix_XGBoost.png
confusion_matrix_Hybrid_ANN_RF.png
combined_roc_auc_all_models.png
combined_accuracy_curve_all_models.png
all_models_performance_comparison.png
comparison_accuracy.png
comparison_precision_weighted.png
comparison_recall_weighted.png
comparison_f1_weighted.png
comparison_roc_auc_macro_ovr.png
model_metrics.csv
```

The script does not generate separate per-model ROC or accuracy learning curve files. It generates combined ROC-AUC and combined accuracy curves for all models.

`report_figures/` is ignored by Git because these files are generated outputs. Re-run `python evaluation_plots.py` whenever you need to recreate them.

## Vercel Production Deployment

This project includes Vercel production configuration:

```text
vercel.json
.python-version
requirements.txt
```

Production behavior:

- Vercel detects `main.py` and the top-level FastAPI `app`.
- Python runtime version is pinned to `3.12`.
- Runtime dependencies are installed from `requirements.txt`.
- Generated report/evaluation files are excluded from the serverless function bundle.
- The app uses `artifacts/lifestyle_risk_model.joblib` instead of training on serverless cold start.

Deploy checklist:

1. Train locally:

```bash
python life_style_train.py
```

2. Confirm this file exists:

```text
artifacts/lifestyle_risk_model.joblib
```

3. Commit required production files:

```text
main.py
life_style_train.py
requirements.txt
vercel.json
.python-version
templates/
static/
data/Data.xlsx
artifacts/lifestyle_risk_model.joblib
```

4. Deploy with Vercel.

After deployment, check:

```http
GET /api/health
```

Expected production status:

```json
{
  "status": "ok",
  "training_status": "model_ready"
}
```

## API Endpoints

### Website

```http
GET /
```

Loads the one-page website.

### Health Check

```http
GET /api/health
```

Returns server status and training result.

Example:

```json
{
  "status": "ok",
  "training_status": "trained",
  "training_result": {
    "status": "trained",
    "best_model": "SVM",
    "accuracy": 0.8348,
    "f1_weighted": 0.8409
  }
}
```

### Prediction

```http
POST /api/predict
```

Example request body:

```json
{
  "age": 28,
  "gender": "Male",
  "height_cm": 170,
  "weight_kg": 72,
  "fruit_veg_freq": "Sometimes",
  "fast_food_freq": "1-2 times a week",
  "water_8_glasses": "Yes",
  "sugary_drinks_freq": "Occasionally",
  "exercise_freq": "3-4 times a week",
  "exercise_type": "Walking",
  "sleep_hours": "7-8 hours",
  "stress_level": "Moderate",
  "smoke": "No",
  "alcohol": "No",
  "chronic_conditions": "None",
  "family_history_heart": "No",
  "fasting_glucose": "95",
  "overall_health": "Good"
}
```

Example response:

```json
{
  "risk_score": 6,
  "risk_label": "Moderate Risk",
  "model_prediction": "Moderate Risk",
  "bmi": 24.91,
  "bmi_category": "Normal",
  "recommendations": [
    "Add fruits and vegetables most days of the week.",
    "Use stress-management habits such as walking, breathing exercises, or planned breaks."
  ],
  "prediction_source": "trained_model",
  "note": "This is a lifestyle screening estimate, not a medical diagnosis."
}
```

## How Prediction Works

1. The user enters data on the website.
2. `static/app.js` sends the form data to `/api/predict`.
3. `main.py` validates the request with Pydantic.
4. `life_style_train.py` prepares the input row.
5. BMI is calculated from height and weight.
6. The trained model predicts the risk label.
7. Recommendation rules generate practical advice.
8. The frontend displays the result.

## Important Files

- `main.py`: FastAPI app, startup training, API routes
- `life_style_train.py`: dataset cleaning, BMI, training, prediction, recommendations
- `evaluation_plots.py`: generates confusion matrices, combined curves, comparison graphs, and metrics CSV
- `requirements.txt`: runtime dependencies for local server and Vercel production
- `requirements-dev.txt`: optional local evaluation dependencies
- `vercel.json`: Vercel bundle configuration
- `.python-version`: Vercel Python version pin
- `templates/index.html`: website form and result area
- `static/app.js`: browser-side form submission and refresh button
- `static/styles.css`: page design
- `data/Data.xlsx`: final training dataset
- `artifacts/lifestyle_risk_model.joblib`: saved trained model

## Troubleshooting

### `ModuleNotFoundError`

Install dependencies:

```bash
pip install -r requirements.txt
```

### XGBoost install error

Install development dependencies with Python 3.11 or 3.12:

```bash
pip install -r requirements-dev.txt
```

The production app can still run without XGBoost because XGBoost is optional in the training code.

### Dataset not found

Make sure the file exists:

```text
data/Data.xlsx
```

### Server already running

Stop the old server process or run on another port:

```bash
python -m uvicorn main:app --reload --port 8001
```

### Vercel says model is missing

Run training locally and commit:

```text
artifacts/lifestyle_risk_model.joblib
```

## Note

This system is a lifestyle screening and educational tool. It is not a medical diagnosis. Users with high-risk results should consult a qualified healthcare professional.
