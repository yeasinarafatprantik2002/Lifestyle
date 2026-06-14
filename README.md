# Lifestyle Health Risk Predictor

FastAPI project for collecting lifestyle data from a one-page website, calculating BMI, predicting lifestyle health risk, and showing recommendations.

## Run the Website

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

Open:

```text
http://127.0.0.1:8000
```

## Add Training Data Later

Place this Excel file in the `data` folder:

```text
data/Data.xlsx
```

The FastAPI server automatically trains when it starts:

```bash
uvicorn main:app --reload
```

You can also train manually:

```bash
python life_style_train.py
```

That creates:

```text
artifacts/lifestyle_risk_model.joblib
data/lifestyle_health_risk_labeled_dataset.csv
data/model_results.csv
```

When the model file exists, the website API will use it for prediction. Until then, it uses the risk-scoring logic in `life_style_train.py`.
