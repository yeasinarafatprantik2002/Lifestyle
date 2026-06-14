"""
Generate evaluation visualizations for the Lifestyle Health Risk Predictor.

Run from the project root:
    python evaluation_plots.py

Outputs are saved in:
    report_figures/evaluation/

Generated curve files are combined across all models. Individual per-model
ROC-AUC and accuracy curve images are not generated.
"""

from pathlib import Path
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
    roc_auc_score,
)
from sklearn.model_selection import learning_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import label_binarize
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from life_style_train import (
    FEATURE_COLS,
    INPUT_FILE,
    clean_dataset,
    make_preprocessor,
    risk_label,
    risk_score_row,
    split_training_data,
)

try:
    from xgboost import XGBClassifier

    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False


warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "report_figures" / "evaluation"
OUT_DIR.mkdir(parents=True, exist_ok=True)


class XGBLabelEncodedClassifier(BaseEstimator, ClassifierMixin):
    """Use XGBoost in a sklearn pipeline while keeping string class labels."""

    def __init__(
        self,
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state

    def fit(self, X, y):
        self.classes_ = np.array(sorted(pd.Series(y).unique()))
        self.class_to_int_ = {label: index for index, label in enumerate(self.classes_)}
        y_encoded = np.array([self.class_to_int_[label] for label in y])

        self.model_ = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            objective="multi:softprob",
            num_class=len(self.classes_),
            eval_metric="mlogloss",
            random_state=self.random_state,
        )
        self.model_.fit(X, y_encoded)
        return self

    def predict(self, X):
        predictions = self.model_.predict(X).astype(int)
        return self.classes_[predictions]

    def predict_proba(self, X):
        return self.model_.predict_proba(X)


def build_models(train_size):
    neighbors = max(1, min(7, train_size))

    models = {
        "KNN": KNeighborsClassifier(n_neighbors=neighbors),
        "Decision_Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
        "Random_Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced",
        ),
        "Logistic_Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
        ),
        "SVM": SVC(
            kernel="rbf",
            class_weight="balanced",
            probability=True,
            random_state=42,
        ),
        "ANN": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=800,
            random_state=42,
        ),
    }

    if HAS_XGBOOST:
        models["XGBoost"] = XGBLabelEncodedClassifier()

    models["Hybrid_ANN_RF"] = VotingClassifier(
        estimators=[
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=200,
                    random_state=42,
                    class_weight="balanced",
                ),
            ),
            (
                "ann",
                MLPClassifier(
                    hidden_layer_sizes=(64, 32),
                    max_iter=800,
                    random_state=42,
                ),
            ),
        ],
        voting="soft",
    )

    return models


def make_model_pipeline(model):
    return Pipeline([
        ("prep", make_preprocessor()),
        ("model", model),
    ])


def safe_roc_auc(pipe, X_test, y_test, classes):
    try:
        y_score = pipe.predict_proba(X_test)

        if len(classes) == 2:
            return roc_auc_score(y_test, y_score[:, 1])

        y_binary = label_binarize(y_test, classes=classes)
        return roc_auc_score(y_binary, y_score, average="macro", multi_class="ovr")
    except Exception:
        return np.nan


def macro_roc_curve_data(pipe, X_test, y_test, classes):
    y_score = pipe.predict_proba(X_test)

    if len(classes) == 2:
        y_binary = label_binarize(y_test, classes=classes).ravel()
        fpr, tpr, _ = roc_curve(y_binary, y_score[:, 1])
        return fpr, tpr, auc(fpr, tpr)

    y_binary = label_binarize(y_test, classes=classes)
    fpr_by_class = {}
    tpr_by_class = {}

    for index in range(len(classes)):
        fpr_by_class[index], tpr_by_class[index], _ = roc_curve(
            y_binary[:, index],
            y_score[:, index],
        )

    all_fpr = np.unique(np.concatenate([fpr_by_class[index] for index in range(len(classes))]))
    mean_tpr = np.zeros_like(all_fpr)

    for index in range(len(classes)):
        mean_tpr += np.interp(all_fpr, fpr_by_class[index], tpr_by_class[index])

    mean_tpr /= len(classes)
    return all_fpr, mean_tpr, auc(all_fpr, mean_tpr)


def plot_confusion_matrix(pipe, X_test, y_test, model_name):
    try:
        predictions = pipe.predict(X_test)
        fig, ax = plt.subplots(figsize=(7, 5))
        ConfusionMatrixDisplay.from_predictions(
            y_test,
            predictions,
            cmap="Blues",
            values_format="d",
            ax=ax,
        )
        ax.set_title(f"Confusion Matrix - {model_name.replace('_', ' ')}")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        fig.tight_layout()

        output_path = OUT_DIR / f"confusion_matrix_{model_name}.png"
        fig.savefig(output_path, dpi=300)
        plt.close(fig)
        return output_path
    except Exception as exc:
        print(f"Confusion matrix skipped for {model_name}: {exc}")
        return None


def calculate_accuracy_curve_data(pipe, X, y, model_name):
    try:
        class_counts = pd.Series(y).value_counts()
        cv_folds = int(min(3, class_counts.min()))
        if cv_folds < 2:
            raise ValueError("Not enough samples per class for cross-validation.")

        train_sizes, train_scores, test_scores = learning_curve(
            pipe,
            X,
            y,
            cv=cv_folds,
            train_sizes=np.linspace(0.2, 1.0, 5),
            scoring="accuracy",
            n_jobs=-1,
        )

        return train_sizes, test_scores.mean(axis=1)
    except Exception as exc:
        print(f"Accuracy curve data skipped for {model_name}: {exc}")
        return None, None


def plot_combined_roc_curves(trained_pipelines, X_test, y_test, classes):
    fig, ax = plt.subplots(figsize=(9, 7))
    plotted = False

    for model_name, pipe in trained_pipelines.items():
        try:
            fpr, tpr, roc_auc = macro_roc_curve_data(pipe, X_test, y_test, classes)
            ax.plot(
                fpr,
                tpr,
                linewidth=2,
                label=f"{model_name.replace('_', ' ')} (AUC={roc_auc:.3f})",
            )
            plotted = True
        except Exception as exc:
            print(f"Combined ROC skipped for {model_name}: {exc}")

    if not plotted:
        plt.close(fig)
        return None

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    ax.set_title("Combined ROC-AUC Curve - All Models")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()

    output_path = OUT_DIR / "combined_roc_auc_all_models.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def plot_combined_accuracy_curves(accuracy_curve_data):
    fig, ax = plt.subplots(figsize=(9, 7))
    plotted = False

    for model_name, curve_data in accuracy_curve_data.items():
        train_sizes, validation_scores = curve_data
        if train_sizes is None or validation_scores is None:
            continue

        ax.plot(
            train_sizes,
            validation_scores,
            marker="o",
            linewidth=2,
            label=model_name.replace("_", " "),
        )
        plotted = True

    if not plotted:
        plt.close(fig)
        return None

    ax.set_title("Combined Validation Accuracy Learning Curve - All Models")
    ax.set_xlabel("Training Set Size")
    ax.set_ylabel("Validation Accuracy")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()

    output_path = OUT_DIR / "combined_accuracy_curve_all_models.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def plot_model_comparison(results_df):
    metrics = [
        "Accuracy",
        "Precision_weighted",
        "Recall_weighted",
        "F1_weighted",
        "ROC_AUC_macro_ovr",
    ]

    comparison_df = results_df[["Model"] + metrics].copy().set_index("Model")

    fig, ax = plt.subplots(figsize=(12, 6))
    comparison_df.plot(kind="bar", ax=ax)
    ax.set_title("All Models Performance Comparison")
    ax.set_xlabel("Machine Learning Model")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(title="Metric", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.xticks(rotation=35, ha="right")
    fig.tight_layout()

    combined_path = OUT_DIR / "all_models_performance_comparison.png"
    fig.savefig(combined_path, dpi=300)
    plt.close(fig)

    metric_paths = {}
    for metric in metrics:
        metric_df = results_df[["Model", metric]].sort_values(metric, ascending=False)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(metric_df["Model"], metric_df[metric])
        ax.set_title(f"All Models {metric.replace('_', ' ')} Comparison")
        ax.set_xlabel("Machine Learning Model")
        ax.set_ylabel(metric.replace("_", " "))
        ax.set_ylim(0, 1.05)
        ax.grid(True, axis="y", alpha=0.3)
        plt.xticks(rotation=35, ha="right")
        fig.tight_layout()

        output_path = OUT_DIR / f"comparison_{metric.lower()}.png"
        fig.savefig(output_path, dpi=300)
        plt.close(fig)
        metric_paths[metric] = output_path

    return combined_path, metric_paths


def load_training_data():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {INPUT_FILE}")

    df = pd.read_excel(INPUT_FILE)
    df = clean_dataset(df)
    df["risk_score"] = df.apply(risk_score_row, axis=1)
    df["risk_label"] = df["risk_score"].map(risk_label)
    return df


def main():
    df = load_training_data()
    X = df[FEATURE_COLS]
    y = df["risk_label"]

    X_train, X_test, y_train, y_test = split_training_data(X, y)
    classes = np.array(sorted(y.unique()))
    models = build_models(len(X_train))

    rows = []
    trained_pipelines = {}
    accuracy_curve_data = {}
    for model_name, model in models.items():
        print(f"Training and evaluating: {model_name}")
        pipe = make_model_pipeline(model)

        try:
            pipe.fit(X_train, y_train)
            predictions = pipe.predict(X_test)
        except Exception as exc:
            print(f"Model skipped for {model_name}: {exc}")
            continue

        trained_pipelines[model_name] = pipe

        confusion_matrix_path = plot_confusion_matrix(pipe, X_test, y_test, model_name)
        train_sizes, validation_scores = calculate_accuracy_curve_data(
            pipe,
            X,
            y,
            model_name,
        )
        accuracy_curve_data[model_name] = (train_sizes, validation_scores)

        rows.append({
            "Model": model_name.replace("_", " "),
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision_weighted": precision_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0,
            ),
            "Recall_weighted": recall_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0,
            ),
            "F1_weighted": f1_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0,
            ),
            "ROC_AUC_macro_ovr": safe_roc_auc(pipe, X_test, y_test, classes),
            "Confusion_Matrix_File": str(confusion_matrix_path or ""),
            "ROC_AUC_File": "",
            "Accuracy_Curve_File": "",
        })

    if not rows:
        raise RuntimeError("No models were evaluated successfully.")

    results_df = pd.DataFrame(rows).sort_values(
        ["Accuracy", "F1_weighted"],
        ascending=False,
    )

    comparison_path, metric_paths = plot_model_comparison(results_df)
    combined_roc_path = plot_combined_roc_curves(trained_pipelines, X_test, y_test, classes)
    combined_accuracy_path = plot_combined_accuracy_curves(accuracy_curve_data)

    results_df["ROC_AUC_File"] = str(combined_roc_path or "")
    results_df["Accuracy_Curve_File"] = str(combined_accuracy_path or "")

    metrics_path = OUT_DIR / "model_metrics.csv"
    results_df.to_csv(metrics_path, index=False)

    print("\nSaved evaluation outputs in:")
    print(OUT_DIR)
    print("\nMain comparison graph:")
    print(comparison_path)
    print("\nIndividual comparison graphs:")
    for metric, path in metric_paths.items():
        print(f"- {metric}: {path}")
    print("\nCombined curve graphs:")
    print(f"- ROC-AUC: {combined_roc_path or 'Not generated'}")
    print(f"- Accuracy: {combined_accuracy_path or 'Not generated'}")
    print("\nMetrics CSV:")
    print(metrics_path)
    print("\nFinal metrics table:")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
