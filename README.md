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
|-- life_style_train.py
|-- main.py
|-- requirements.txt
|-- .gitignore
`-- README.md
```

## Requirements

- Python 3.11 or newer recommended
- Python 3.14 may work, but some ML libraries such as `xgboost` can have compatibility issues depending on package availability.

## Setup

From the project folder:

```bash
pip install -r requirements.txt
```

If `xgboost` fails to install, you can remove `xgboost` from `requirements.txt`. The project will still run because XGBoost is optional in `life_style_train.py`.

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

When the server starts, it automatically trains the model using:

```text
data/Data.xlsx
```

After training, it saves:

```text
artifacts/lifestyle_risk_model.joblib
data/lifestyle_health_risk_labeled_dataset.csv
data/model_results.csv
```

## Manual Training

You can also train without starting the website:

```bash
python life_style_train.py
```

This will clean the dataset, calculate BMI, create risk labels, train multiple models, save the best model, and write result files.

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

Use Python 3.11 or 3.12, or remove `xgboost` from `requirements.txt`. The app can still train and predict without XGBoost.

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

## Note

This system is a lifestyle screening and educational tool. It is not a medical diagnosis. Users with high-risk results should consult a qualified healthcare professional.
