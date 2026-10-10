from pathlib import Path

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
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# --------------------------------------------------
# 1. Paths and configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
OUTPUT_DIR = ROOT / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42


# --------------------------------------------------
# 2. Load and clean dataset
# --------------------------------------------------

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found at: {DATA_PATH}\n"
        "Place the Telco Customer Churn CSV inside the data folder."
    )

df = pd.read_csv(DATA_PATH)

required_columns = {"customerID", "Churn", "TotalCharges"}
missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

df = df.drop_duplicates().copy()

# Convert blank TotalCharges values to NaN, then numeric.
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

# Remove the identifier; it is not a useful predictive feature.
X = df.drop(columns=["customerID", "Churn"])
y = df["Churn"].map({"No": 0, "Yes": 1})

if y.isna().any():
    raise ValueError("Unexpected values found in the Churn column.")

print("=" * 60)
print("CUSTOMER CHURN PREDICTION")
print("=" * 60)
print(f"Dataset shape: {df.shape}")
print(f"Customers: {len(df):,}")
print(f"Churn rate: {y.mean():.2%}")
print("\nClass distribution:")
print(y.map({0: "No Churn", 1: "Churn"}).value_counts())


# --------------------------------------------------
# 3. Identify numeric and categorical features
# --------------------------------------------------

numeric_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "category", "bool"]
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


# --------------------------------------------------
# 4. Train-test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y,
)

print(f"\nTraining samples: {len(X_train):,}")
print(f"Testing samples: {len(X_test):,}")


# --------------------------------------------------
# 5. Train and compare models
# --------------------------------------------------

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
        n_jobs=-1,
    ),
}

results = []
trained_models = {}

for name, classifier in models.items():
    print(f"\nTraining {name}...")

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    roc_auc = roc_auc_score(y_test, probabilities)

    trained_models[name] = pipeline

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "ROC-AUC": roc_auc,
    })

    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC-AUC:  {roc_auc:.4f}")

results_df = pd.DataFrame(results).sort_values(
    "ROC-AUC", ascending=False
)

print("\nModel comparison:")
print(results_df.to_string(index=False))

best_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_name]

print(f"\nBest model by test ROC-AUC: {best_name}")


# --------------------------------------------------
# 6. Evaluate the selected model
# --------------------------------------------------

predictions = best_model.predict(X_test)
probabilities = best_model.predict_proba(X_test)[:, 1]

report = classification_report(
    y_test,
    predictions,
    target_names=["No Churn", "Churn"],
    zero_division=0,
)

print("\nClassification report:")
print(report)

with open(
    OUTPUT_DIR / "churn_analysis_report.txt",
    "w",
    encoding="utf-8",
) as file:
    file.write("CUSTOMER CHURN ANALYSIS REPORT\n")
    file.write("=" * 40 + "\n\n")
    file.write(f"Dataset rows: {len(df)}\n")
    file.write(f"Training samples: {len(X_train)}\n")
    file.write(f"Testing samples: {len(X_test)}\n")
    file.write(f"Best model: {best_name}\n")
    file.write(f"Accuracy: {accuracy_score(y_test, predictions):.4f}\n")
    file.write(f"ROC-AUC: {roc_auc_score(y_test, probabilities):.4f}\n\n")
    file.write("MODEL COMPARISON\n")
    file.write(results_df.to_string(index=False))
    file.write("\n\nCLASSIFICATION REPORT\n")
    file.write(report)


# --------------------------------------------------
# 7. Save model and test predictions
# --------------------------------------------------

model_path = OUTPUT_DIR / "customer_churn_model.joblib"

joblib.dump(best_model, model_path)

test_results = X_test.copy()
test_results["ActualChurn"] = y_test.to_numpy()
test_results["PredictedChurn"] = predictions
test_results["ChurnProbability"] = probabilities

test_results.to_csv(
    OUTPUT_DIR / "churn_predictions.csv",
    index=False,
)


# --------------------------------------------------
# 8. Generate visualizations
# --------------------------------------------------

sns.set_theme(style="whitegrid")

# Churn distribution
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
plt.savefig(OUTPUT_DIR / "churn_distribution.png", dpi=200)
plt.close()

# Confusion matrix
cm = confusion_matrix(y_test, predictions)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No Churn", "Churn"],
    yticklabels=["No Churn", "Churn"],
)
plt.title(f"Confusion Matrix — {best_name}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=200)
plt.close()

# ROC curve
fpr, tpr, _ = roc_curve(y_test, probabilities)
auc = roc_auc_score(y_test, probabilities)

plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f"{best_name} (AUC = {auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", label="Random Classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "roc_curve.png", dpi=200)
plt.close()

# Model comparison
plt.figure(figsize=(8, 5))
sns.barplot(
    data=results_df,
    x="Model",
    y="ROC-AUC",
)
plt.ylim(0, 1)
plt.title("Model Comparison by ROC-AUC")
plt.xticks(rotation=10)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "model_comparison.png", dpi=200)
plt.close()

print("\n" + "=" * 60)
print("TRAINING COMPLETED")
print("=" * 60)
print(f"Saved model: {model_path}")
print(f"Saved report: {OUTPUT_DIR / 'churn_analysis_report.txt'}")
print(f"Saved predictions: {OUTPUT_DIR / 'churn_predictions.csv'}")
print(f"Saved visualizations in: {OUTPUT_DIR}")