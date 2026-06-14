import re
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "data/Data.xlsx"
MODEL_FILE = BASE_DIR / "artifacts/lifestyle_risk_model.joblib"
LABELED_DATASET_FILE = BASE_DIR / "data/lifestyle_health_risk_labeled_dataset.csv"
MODEL_RESULTS_FILE = BASE_DIR / "data/model_results.csv"

FEATURE_COLS = [
    "age", "gender", "height_cm", "weight_kg", "bmi",
    "fruit_veg_freq", "fast_food_freq", "water_8_glasses", "sugary_drinks_freq",
    "exercise_freq", "exercise_type", "sleep_hours", "stress_level",
    "smoke", "alcohol", "chronic_conditions", "family_history_heart",
    "fasting_glucose_mgdl", "overall_health"
]

NUMERIC_FEATURES = ["age", "height_cm", "weight_kg", "bmi", "sleep_hours", "fasting_glucose_mgdl"]
CATEGORICAL_FEATURES = [c for c in FEATURE_COLS if c not in NUMERIC_FEATURES]


def normalize_text(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    s = s.replace("’", "'").replace("–", "-").replace("—", "-")
    s = re.sub(r"\s+", " ", s)
    if s.lower() in {
        "", ".", "n/a", "na", "unknown", "not sure", "not know", "not known",
        "dont know", "don't know", "i dont know", "i don't know", "dk", "idk"
    }:
        return np.nan
    return s


def parse_sleep(x):
    if pd.isna(x):
        return np.nan
    s = str(x).lower().replace("â€“", "-").replace("â€”", "-")
    s = re.sub(r"[\u2012\u2013\u2014\u2212]", "-", s)
    if "less than 5" in s:
        return 4.5
    if "5-6" in s or "5–6" in s or "below 6" in s:
        return 5.5
    if "7-8" in s or "7–8" in s:
        return 7.5
    if "more than 8" in s:
        return 8.5
    m = re.search(r"(\d+(\.\d+)?)", s)
    return float(m.group(1)) if m else np.nan


def parse_glucose(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    m = re.search(r"(\d+(\.\d+)?)", s)
    if not m:
        return np.nan
    val = float(m.group(1))
    if val < 20:
        return round(val * 18, 1)
    return val


def clean_dataset(df):
    if len(df.columns) < 19:
        raise ValueError("Training dataset must contain at least 19 columns.")

    rename_map = {
        df.columns[0]: "timestamp",
        df.columns[1]: "age",
        df.columns[2]: "gender",
        df.columns[3]: "height_cm",
        df.columns[4]: "weight_kg",
        df.columns[5]: "fruit_veg_freq",
        df.columns[6]: "fast_food_freq",
        df.columns[7]: "water_8_glasses",
        df.columns[8]: "sugary_drinks_freq",
        df.columns[9]: "exercise_freq",
        df.columns[10]: "exercise_type",
        df.columns[11]: "sleep_hours_text",
        df.columns[12]: "stress_level",
        df.columns[13]: "smoke",
        df.columns[14]: "alcohol",
        df.columns[15]: "chronic_conditions",
        df.columns[16]: "family_history_heart",
        df.columns[17]: "fasting_glucose",
        df.columns[18]: "overall_health",
    }
    df = df.rename(columns=rename_map).copy()

    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].map(normalize_text)

    for col in ["age", "height_cm", "weight_kg"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["sleep_hours"] = df["sleep_hours_text"].map(parse_sleep)
    df["fasting_glucose_mgdl"] = df["fasting_glucose"].map(parse_glucose)
    height_m = df["height_cm"] / 100
    df["bmi"] = np.where(height_m > 0, df["weight_kg"] / (height_m ** 2), np.nan)
    return df


def row_value(row, key, default=np.nan):
    if hasattr(row, "get"):
        return row.get(key, default)
    return row[key] if key in row else default


def risk_score_row(r):
    score = 0

    age = row_value(r, "age")
    if pd.notna(age):
        if age >= 45:
            score += 2
        elif age >= 30:
            score += 1

    bmi = row_value(r, "bmi")
    if pd.notna(bmi):
        if bmi >= 30:
            score += 3
        elif bmi >= 25:
            score += 2
        elif bmi < 18.5:
            score += 1

    fv_raw = row_value(r, "fruit_veg_freq")
    fv = str(fv_raw).lower() if pd.notna(fv_raw) else ""
    if fv in {"rarely", "never"}:
        score += 2
    elif fv == "sometimes":
        score += 1

    ff_raw = row_value(r, "fast_food_freq")
    ff = str(ff_raw).lower() if pd.notna(ff_raw) else ""
    if ff in {"daily", "every day", "frequently", "almost every day"}:
        score += 3
    elif ff in {"3-4 times a week", "3-5 times a week"}:
        score += 2
    elif ff in {"1-2 times a week", "1-3 days per week", "occasionally"}:
        score += 1

    water_raw = row_value(r, "water_8_glasses")
    water = str(water_raw).lower() if pd.notna(water_raw) else ""
    if water == "no":
        score += 1

    sugar_raw = row_value(r, "sugary_drinks_freq")
    sugar = str(sugar_raw).lower() if pd.notna(sugar_raw) else ""
    if sugar in {"daily", "every day", "frequently", "almost every day"}:
        score += 2
    elif sugar in {"3-4 times a week", "3-5 times a week", "1-2 times a week", "1-3 days per week", "occasionally"}:
        score += 1

    ex_raw = row_value(r, "exercise_freq")
    ex = str(ex_raw).lower() if pd.notna(ex_raw) else ""
    if ex in {"never", "rarely"}:
        score += 3
    elif ex in {"1-2 times a week", "1-3 days per week"}:
        score += 2
    elif ex in {"3-4 times a week", "3-5 times a week", "4-6 days per week"}:
        score += 1

    sleep_hours = row_value(r, "sleep_hours")
    if pd.notna(sleep_hours):
        if sleep_hours < 5:
            score += 2
        elif sleep_hours < 6:
            score += 1
        elif sleep_hours > 9:
            score += 1

    stress_raw = row_value(r, "stress_level")
    stress = str(stress_raw).lower() if pd.notna(stress_raw) else ""
    if stress == "high":
        score += 2
    elif stress == "moderate":
        score += 1

    smoke_raw = row_value(r, "smoke")
    smoke = str(smoke_raw).lower() if pd.notna(smoke_raw) else ""
    if smoke in {"yes", "daily", "frequently"}:
        score += 3
    elif smoke == "occasionally":
        score += 2

    alcohol_raw = row_value(r, "alcohol")
    alcohol = str(alcohol_raw).lower() if pd.notna(alcohol_raw) else ""
    if alcohol in {"yes", "daily", "frequently"}:
        score += 2
    elif alcohol == "occasionally":
        score += 1

    chronic_raw = row_value(r, "chronic_conditions")
    chronic = str(chronic_raw).lower() if pd.notna(chronic_raw) else ""
    if chronic and chronic not in {"none", "nothing"}:
        items = [p.strip() for p in re.split(r"[;,]", chronic) if p.strip() and p.strip() != "none"]
        score += min(4, max(1, len(items) * 2))

    fam_raw = row_value(r, "family_history_heart")
    fam = str(fam_raw).lower() if pd.notna(fam_raw) else ""
    if fam == "yes":
        score += 2

    glu = row_value(r, "fasting_glucose_mgdl")
    if pd.notna(glu):
        if glu >= 126:
            score += 4
        elif glu >= 100:
            score += 2

    health_raw = row_value(r, "overall_health")
    health = str(health_raw).lower() if pd.notna(health_raw) else ""
    if health == "poor":
        score += 3
    elif health == "fair":
        score += 2
    elif health == "good":
        score += 1

    return score


def risk_label(score):
    if score >= 10:
        return "High Risk"
    if score >= 5:
        return "Moderate Risk"
    return "Low Risk"


def calculate_bmi(height_cm, weight_kg):
    height_cm = pd.to_numeric(height_cm, errors="coerce")
    weight_kg = pd.to_numeric(weight_kg, errors="coerce")
    if pd.isna(height_cm) or pd.isna(weight_kg) or height_cm <= 0:
        return np.nan
    return round(float(weight_kg) / ((float(height_cm) / 100) ** 2), 2)


def make_preprocessor():
    return ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), NUMERIC_FEATURES),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), CATEGORICAL_FEATURES),
    ])


def split_training_data(X, y):
    if len(X) < 2:
        raise ValueError("Training requires at least two rows.")
    if y.nunique() < 2:
        raise ValueError("Training requires at least two risk classes.")

    class_counts = y.value_counts()
    stratify = y if class_counts.min() >= 2 else None
    test_size = 0.2

    if stratify is not None:
        min_test_count = y.nunique()
        test_size = max(test_size, min_test_count / len(y))

    return train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=stratify
    )


def build_model_candidates(train_size):
    neighbors = max(1, min(7, train_size))
    return {
        "KNN": KNeighborsClassifier(n_neighbors=neighbors),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced"),
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "SVM": SVC(kernel="rbf", class_weight="balanced", random_state=42),
        "ANN": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=800, random_state=42),
    }


def train_pipeline(model, X_train, y_train):
    pipe = Pipeline([
        ("prep", make_preprocessor()),
        ("model", model),
    ])
    pipe.fit(X_train, y_train)
    return pipe


def bmi_category(bmi):
    if pd.isna(bmi):
        return "Unknown"
    if bmi < 18.5:
        return "Underweight"
    if bmi < 25:
        return "Normal"
    if bmi < 30:
        return "Overweight"
    return "Obese"


def recommendation_list(row, label):
    recommendations = []
    bmi = row.get("bmi")

    if pd.notna(bmi):
        if bmi >= 25:
            recommendations.append("Work toward a healthy weight with regular activity and balanced portions.")
        elif bmi < 18.5:
            recommendations.append("Increase nutrient-dense meals and consider checking for causes of low weight.")

    if str(row.get("fruit_veg_freq", "")).lower() in {"rarely", "never", "sometimes"}:
        recommendations.append("Add fruits and vegetables most days of the week.")

    if str(row.get("fast_food_freq", "")).lower() in {"daily", "every day", "frequently", "almost every day", "3-4 times a week", "3-5 times a week"}:
        recommendations.append("Reduce fast food and choose home-cooked meals when possible.")

    if str(row.get("water_8_glasses", "")).lower() == "no":
        recommendations.append("Drink more water through the day and limit sweet drinks.")

    if str(row.get("sugary_drinks_freq", "")).lower() in {"daily", "every day", "frequently", "almost every day"}:
        recommendations.append("Replace sugary drinks with water, unsweetened tea, or low-sugar options.")

    if str(row.get("exercise_freq", "")).lower() in {"never", "rarely", "1-2 times a week", "1-3 days per week"}:
        recommendations.append("Build up to at least 150 minutes of moderate exercise per week.")

    sleep_hours = row.get("sleep_hours")
    if pd.notna(sleep_hours) and (sleep_hours < 6 or sleep_hours > 9):
        recommendations.append("Aim for a consistent sleep routine around 7-8 hours per night.")

    if str(row.get("stress_level", "")).lower() in {"moderate", "high"}:
        recommendations.append("Use stress-management habits such as walking, breathing exercises, or planned breaks.")

    if str(row.get("smoke", "")).lower() in {"yes", "daily", "frequently", "occasionally"}:
        recommendations.append("Avoid smoking and seek support if quitting feels difficult.")

    if str(row.get("alcohol", "")).lower() in {"yes", "daily", "frequently", "occasionally"}:
        recommendations.append("Limit alcohol intake and keep alcohol-free days each week.")

    glucose = row.get("fasting_glucose_mgdl")
    if pd.notna(glucose) and glucose >= 100:
        recommendations.append("Monitor fasting glucose and discuss the result with a healthcare professional.")

    if label == "High Risk":
        recommendations.append("Book a medical check-up for a personalized assessment.")

    if not recommendations:
        recommendations.append("Keep your current healthy habits and repeat screening regularly.")

    return recommendations


def load_trained_model():
    if not MODEL_FILE.exists():
        return None
    try:
        return load(MODEL_FILE)
    except Exception:
        return None


def build_prediction_row(payload):
    row = {
        "age": pd.to_numeric(payload.get("age"), errors="coerce"),
        "gender": normalize_text(payload.get("gender")),
        "height_cm": pd.to_numeric(payload.get("height_cm"), errors="coerce"),
        "weight_kg": pd.to_numeric(payload.get("weight_kg"), errors="coerce"),
        "fruit_veg_freq": normalize_text(payload.get("fruit_veg_freq")),
        "fast_food_freq": normalize_text(payload.get("fast_food_freq")),
        "water_8_glasses": normalize_text(payload.get("water_8_glasses")),
        "sugary_drinks_freq": normalize_text(payload.get("sugary_drinks_freq")),
        "exercise_freq": normalize_text(payload.get("exercise_freq")),
        "exercise_type": normalize_text(payload.get("exercise_type")),
        "sleep_hours_text": normalize_text(payload.get("sleep_hours")),
        "stress_level": normalize_text(payload.get("stress_level")),
        "smoke": normalize_text(payload.get("smoke")),
        "alcohol": normalize_text(payload.get("alcohol")),
        "chronic_conditions": normalize_text(payload.get("chronic_conditions")),
        "family_history_heart": normalize_text(payload.get("family_history_heart")),
        "fasting_glucose": normalize_text(payload.get("fasting_glucose")),
        "overall_health": normalize_text(payload.get("overall_health")),
    }
    row["sleep_hours"] = parse_sleep(row["sleep_hours_text"])
    row["fasting_glucose_mgdl"] = parse_glucose(row["fasting_glucose"])
    row["bmi"] = calculate_bmi(row["height_cm"], row["weight_kg"])
    return row


def predict_from_input(payload):
    row = build_prediction_row(payload)
    score = risk_score_row(row)
    label = risk_label(score)
    model_label = None

    model = load_trained_model()
    if model is not None:
        try:
            input_df = pd.DataFrame([{col: row.get(col, np.nan) for col in FEATURE_COLS}])
            model_label = str(model.predict(input_df)[0])
            label = model_label
        except Exception:
            model_label = None

    return {
        "risk_score": int(score),
        "risk_label": label,
        "model_prediction": model_label,
        "bmi": row["bmi"] if pd.notna(row["bmi"]) else None,
        "bmi_category": bmi_category(row["bmi"]),
        "recommendations": recommendation_list(row, label),
        "prediction_source": "trained_model" if model_label else "rule_based",
        "note": "This is a lifestyle screening estimate, not a medical diagnosis.",
    }


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Training file not found: {INPUT_FILE}")

    df = pd.read_excel(INPUT_FILE)
    df = clean_dataset(df)

    df["risk_score"] = df.apply(risk_score_row, axis=1)
    df["risk_label"] = df["risk_score"].map(risk_label)

    X = df[FEATURE_COLS]
    y = df["risk_label"]

    X_train, X_test, y_train, y_test = split_training_data(X, y)
    models = build_model_candidates(len(X_train))

    results = []
    best_pipeline = None
    best_model_name = None
    best_score = (-1, -1)

    for name, model in models.items():
        pipe = train_pipeline(model, X_train, y_train)
        pred = pipe.predict(X_test)
        metrics = {
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision_weighted": precision_score(y_test, pred, average="weighted", zero_division=0),
            "Recall_weighted": recall_score(y_test, pred, average="weighted", zero_division=0),
            "F1_weighted": f1_score(y_test, pred, average="weighted", zero_division=0),
        }
        results.append(metrics)

        score = (metrics["Accuracy"], metrics["F1_weighted"])
        if score > best_score:
            best_score = score
            best_pipeline = pipe
            best_model_name = name

    if HAS_XGBOOST:
        le = LabelEncoder()
        le.fit(y_train)
        preprocessor = make_preprocessor()
        X_train_p = preprocessor.fit_transform(X_train)
        X_test_p = preprocessor.transform(X_test)

        xgb = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softmax",
            num_class=len(le.classes_),
            eval_metric="mlogloss",
            random_state=42,
        )
        xgb.fit(X_train_p, le.transform(y_train))
        pred = le.inverse_transform(xgb.predict(X_test_p))
        results.append({
            "Model": "XGBoost",
            "Accuracy": accuracy_score(y_test, pred),
            "Precision_weighted": precision_score(y_test, pred, average="weighted", zero_division=0),
            "Recall_weighted": recall_score(y_test, pred, average="weighted", zero_division=0),
            "F1_weighted": f1_score(y_test, pred, average="weighted", zero_division=0),
        })

    hybrid = VotingClassifier(
        estimators=[
            ("rf", RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")),
            ("ann", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=800, random_state=42)),
        ],
        voting="soft",
    )
    hybrid_pipe = train_pipeline(hybrid, X_train, y_train)
    pred = hybrid_pipe.predict(X_test)
    metrics = {
        "Model": "Hybrid (ANN + RF)",
        "Accuracy": accuracy_score(y_test, pred),
        "Precision_weighted": precision_score(y_test, pred, average="weighted", zero_division=0),
        "Recall_weighted": recall_score(y_test, pred, average="weighted", zero_division=0),
        "F1_weighted": f1_score(y_test, pred, average="weighted", zero_division=0),
    }
    results.append(metrics)

    score = (metrics["Accuracy"], metrics["F1_weighted"])
    if score > best_score:
        best_score = score
        best_pipeline = hybrid_pipe
        best_model_name = "Hybrid (ANN + RF)"

    results_df = pd.DataFrame(results).sort_values(["Accuracy", "F1_weighted"], ascending=False)

    if best_pipeline is None:
        raise RuntimeError("Training completed without a model to save.")

    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    dump(best_pipeline, MODEL_FILE)
    df.to_csv(LABELED_DATASET_FILE, index=False)
    results_df.to_csv(MODEL_RESULTS_FILE, index=False)

    print("Risk label distribution:")
    print(df["risk_label"].value_counts())
    print("\nModel results:")
    print(results_df.to_string(index=False))
    print("\nSaved:")
    print(f"- {LABELED_DATASET_FILE}")
    print(f"- {MODEL_RESULTS_FILE}")
    print(f"- {MODEL_FILE}")
    print(f"\nBest saved model: {best_model_name}")

    return {
        "status": "trained",
        "best_model": best_model_name,
        "accuracy": best_score[0],
        "f1_weighted": best_score[1],
        "model_file": str(MODEL_FILE),
    }


if __name__ == "__main__":
    main()
