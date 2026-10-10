from pathlib import Path
import warnings

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ==================================================
# 1. Configuration
# ==================================================

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
OUTPUT_DIR = ROOT / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.20
CV_FOLDS = 5

sns.set_theme(style="whitegrid")


# ==================================================
# 2. Load and clean data
# ==================================================

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}\n"
        "Place the Telco Customer Churn CSV in the data folder."
    )

df = pd.read_csv(DATA_PATH)
df = df.drop_duplicates().copy()

required_columns = {"customerID", "Churn", "TotalCharges"}
missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

X = df.drop(columns=["customerID", "Churn"])
y = df["Churn"].map({"No": 0, "Yes": 1})

if y.isna().any():
    raise ValueError("Unexpected values in the Churn column.")

print("=" * 65)
print("CUSTOMER CHURN PREDICTION — MODEL DEVELOPMENT")
print("=" * 65)
print(f"Dataset shape: {df.shape}")
print(f"Total customers: {len(df):,}")
print(f"Overall churn rate: {y.mean():.2%}")


# ==================================================
# 3. Identify feature types
# ==================================================

numeric_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X.select_dtypes(
include=["object", "str", "category", "bool"]
).columns.tolist()

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features),
])


# ==================================================
# 4. Stratified train / validation / test split
# ==================================================

# First reserve the final test set.
X_development, X_test, y_development, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)

# Split the remaining 80% equally into train and validation.
# Final proportions: 60% train, 20% validation, 20% test.
validation_fraction_of_development = (
    VALIDATION_SIZE / (1 - TEST_SIZE)
)

X_train, X_validation, y_train, y_validation = train_test_split(
    X_development,
    y_development,
    test_size=validation_fraction_of_development,
    random_state=RANDOM_STATE,
    stratify=y_development,
)

print("\nDATA SPLIT")
print("-" * 40)
print(f"Training samples:   {len(X_train):,}")
print(f"Validation samples: {len(X_validation):,}")
print(f"Test samples:       {len(X_test):,}")


# ==================================================
# 5. Define candidate models
# ==================================================

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=1,
    ),
}

cv = StratifiedKFold(
    n_splits=CV_FOLDS,
    shuffle=True,
    random_state=RANDOM_STATE,
)

scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}

cv_results = []
candidate_pipelines = {}

# ==================================================
# 6. Cross-validation and model selection
# ==================================================

print("\nFIVE-FOLD STRATIFIED CROSS-VALIDATION")
print("-" * 65)

for name, classifier in models.items():
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])

    scores = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=False,
    )

    row = {"Model": name}

    for metric in scoring:
        values = scores[f"test_{metric}"]
        row[f"CV {metric.upper()} Mean"] = values.mean()
        row[f"CV {metric.upper()} Std"] = values.std()

    cv_results.append(row)
    candidate_pipelines[name] = pipeline

    print(f"\n{name}")
    print(
        f"ROC-AUC: {row['CV ROC_AUC Mean']:.4f} "
        f"+/- {row['CV ROC_AUC Std']:.4f}"
    )
    print(
        f"F1-score: {row['CV F1 Mean']:.4f} "
        f"+/- {row['CV F1 Std']:.4f}"
    )

cv_results_df = pd.DataFrame(cv_results).sort_values(
    "CV ROC_AUC Mean",
    ascending=False,
)

cv_results_df.to_csv(
    OUTPUT_DIR / "cross_validation_results.csv",
    index=False,
)

best_name = cv_results_df.iloc[0]["Model"]
best_model = candidate_pipelines[best_name]

print("\nCROSS-VALIDATION COMPARISON")
print(cv_results_df.to_string(index=False))
print(f"\nSelected model: {best_name}")


# ==================================================
# 7. Fit selected model on training data
# ==================================================

best_model.fit(X_train, y_train)

validation_probabilities = best_model.predict_proba(
    X_validation
)[:, 1]

validation_predictions = best_model.predict(X_validation)

validation_metrics = {
    "Accuracy": accuracy_score(
        y_validation, validation_predictions
    ),
    "Precision": precision_score(
        y_validation, validation_predictions, zero_division=0
    ),
    "Recall": recall_score(
        y_validation, validation_predictions, zero_division=0
    ),
    "F1-score": f1_score(
        y_validation, validation_predictions, zero_division=0
    ),
    "ROC-AUC": roc_auc_score(
        y_validation, validation_probabilities
    ),
}

print("\nVALIDATION RESULTS")
print("-" * 40)

for metric, value in validation_metrics.items():
    print(f"{metric}: {value:.4f}")


# ==================================================
# 8. Final test evaluation
# ==================================================

test_probabilities = best_model.predict_proba(X_test)[:, 1]
test_predictions = best_model.predict(X_test)

test_metrics = {
    "Accuracy": accuracy_score(y_test, test_predictions),
    "Precision": precision_score(
        y_test, test_predictions, zero_division=0
    ),
    "Recall": recall_score(
        y_test, test_predictions, zero_division=0
    ),
    "F1-score": f1_score(
        y_test, test_predictions, zero_division=0
    ),
    "ROC-AUC": roc_auc_score(y_test, test_probabilities),
}

print("\nFINAL TEST RESULTS")
print("-" * 40)

for metric, value in test_metrics.items():
    print(f"{metric}: {value:.4f}")

report = classification_report(
    y_test,
    test_predictions,
    labels=[0, 1],
    target_names=["No Churn", "Churn"],
    zero_division=0,
)

print("\nCLASSIFICATION REPORT")
print(report)


# ==================================================
# 9. Save final model
# ==================================================

model_path = OUTPUT_DIR / "customer_churn_model.joblib"

joblib.dump(best_model, model_path)


# ==================================================
# 10. Save test predictions
# ==================================================

test_results = X_test.copy()

test_results.insert(0, "CustomerID", df.loc[X_test.index, "customerID"])
test_results["ActualChurn"] = y_test
test_results["PredictedChurn"] = test_predictions
test_results["ChurnProbability"] = test_probabilities

test_results.to_csv(
    OUTPUT_DIR / "churn_predictions.csv",
    index=False,
)


# ==================================================
# 11. Save evaluation report
# ==================================================

with open(
    OUTPUT_DIR / "churn_analysis_report.txt",
    "w",
    encoding="utf-8",
) as file:
    file.write("CUSTOMER CHURN MODEL EVALUATION\n")
    file.write("=" * 45 + "\n\n")

    file.write(f"Dataset rows: {len(df)}\n")
    file.write(f"Training samples: {len(X_train)}\n")
    file.write(f"Validation samples: {len(X_validation)}\n")
    file.write(f"Test samples: {len(X_test)}\n")
    file.write(f"Cross-validation folds: {CV_FOLDS}\n")
    file.write(f"Selected model: {best_name}\n\n")

    file.write("CROSS-VALIDATION RESULTS\n")
    file.write(cv_results_df.to_string(index=False))
    file.write("\n\nVALIDATION METRICS\n")

    for metric, value in validation_metrics.items():
        file.write(f"{metric}: {value:.4f}\n")

    file.write("\nFINAL TEST METRICS\n")

    for metric, value in test_metrics.items():
        file.write(f"{metric}: {value:.4f}\n")

    file.write("\nCLASSIFICATION REPORT\n")
    file.write(report)


# ==================================================
# 12. Confusion matrix
# ==================================================

cm = confusion_matrix(
    y_test,
    test_predictions,
    labels=[0, 1],
)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No Churn", "Churn"],
    yticklabels=["No Churn", "Churn"],
)

plt.title(f"Test Confusion Matrix — {best_name}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "confusion_matrix.png",
    dpi=200,
)
plt.close()


# ==================================================
# 13. ROC curve
# ==================================================

fpr, tpr, _ = roc_curve(y_test, test_probabilities)

plt.figure(figsize=(7, 5))
plt.plot(
    fpr,
    tpr,
    label=f"{best_name} (AUC = {test_metrics['ROC-AUC']:.3f})",
)
plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random classifier",
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve — Held-Out Test Set")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "roc_curve.png", dpi=200)
plt.close()


# ==================================================
# 14. Cross-validation comparison plot
# ==================================================

plot_data = cv_results_df.sort_values("CV ROC_AUC Mean")

plt.figure(figsize=(8, 5))
plt.barh(
    plot_data["Model"],
    plot_data["CV ROC_AUC Mean"],
    xerr=plot_data["CV ROC_AUC Std"],
    capsize=5,
)

plt.xlim(0, 1)
plt.xlabel("Mean Cross-Validation ROC-AUC")
plt.title("Model Comparison — Five-Fold Cross-Validation")
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "model_comparison.png",
    dpi=200,
)
plt.close()


# ==================================================
# 15. Permutation feature importance
# ==================================================

print("\nCalculating permutation feature importance...")

# Use held-out test rows only for post-selection interpretation.
# ROC-AUC importance is computed from the model's probability scores.
importance = permutation_importance(
    best_model,
    X_test,
    y_test,
    scoring="roc_auc",
    n_repeats=10,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

importance_df = pd.DataFrame({
    "Feature": X_test.columns,
    "ImportanceMean": importance.importances_mean,
    "ImportanceStd": importance.importances_std,
}).sort_values(
    "ImportanceMean",
    ascending=False,
)

importance_df.to_csv(
    OUTPUT_DIR / "permutation_feature_importance.csv",
    index=False,
)

top_features = importance_df.head(15).sort_values(
    "ImportanceMean"
)

plt.figure(figsize=(10, 7))
plt.barh(
    top_features["Feature"],
    top_features["ImportanceMean"],
    xerr=top_features["ImportanceStd"],
    capsize=3,
)

plt.axvline(0, linestyle="--")
plt.xlabel("Decrease in Test ROC-AUC After Shuffling")
plt.title("Permutation Feature Importance — Selected Model")
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "permutation_feature_importance.png",
    dpi=200,
)
plt.close()


# ==================================================
# 16. Churn distribution
# ==================================================

plt.figure(figsize=(7, 5))

sns.countplot(
    x=y.map({0: "No Churn", 1: "Churn"}),
    hue=y.map({0: "No Churn", 1: "Churn"}),
    legend=False,
)

plt.title("Customer Churn Distribution")
plt.xlabel("Customer Status")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "churn_distribution.png",
    dpi=200,
)
plt.close()


# ==================================================
# 17. Summary
# ==================================================

print("\n" + "=" * 65)
print("TRAINING AND EVALUATION COMPLETED")
print("=" * 65)
print(f"Selected model: {best_name}")
print(f"Final test ROC-AUC: {test_metrics['ROC-AUC']:.4f}")
print(f"Final test F1-score: {test_metrics['F1-score']:.4f}")
print(f"Saved model: {model_path}")
print(f"Saved report: {OUTPUT_DIR / 'churn_analysis_report.txt'}")
print(f"Saved cross-validation results: {OUTPUT_DIR / 'cross_validation_results.csv'}")
print(f"Saved feature importance: {OUTPUT_DIR / 'permutation_feature_importance.csv'}")
print(f"Saved visualizations in: {OUTPUT_DIR}")