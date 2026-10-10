from pathlib import Path

import joblib
import pandas as pd


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "outputs" / "customer_churn_model.joblib"
DATA_PATH = (
    BASE_DIR
    / "data"
    / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)


def main():
    # Check that the trained model exists
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}\n"
            "Copy customer_churn_model.joblib from your "
            "Colab outputs ZIP into the outputs folder."
        )

    # Load trained pipeline
    model = joblib.load(MODEL_PATH)

    # Load dataset
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    # Separate features and target
    X = df.drop(
        columns=["Churn", "customerID"],
        errors="ignore"
    )

    # Recreate the features used during training
    X["TotalCharges"] = pd.to_numeric(
        X["TotalCharges"],
        errors="coerce"
    )

    X["TenureGroup"] = pd.cut(
        X["tenure"],
        bins=[-1, 12, 24, 48, float("inf")],
        labels=[
            "New",
            "Developing",
            "Established",
            "Long-term"
        ]
    )

    service_columns = [
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies"
    ]

    def count_active_services(row):
        count = 0

        for column in service_columns:
            value = row[column]

            if value == "Yes":
                count += 1
            elif (
                column == "InternetService"
                and value in ["DSL", "Fiber optic"]
            ):
                count += 1

        return count

    X["ServiceUsageCount"] = X.apply(
        count_active_services,
        axis=1
    )

    # Predict churn
    predictions = model.predict(X)

    probabilities = model.predict_proba(X)[:, 1]

    # Create prediction results
    results = pd.DataFrame({
        "CustomerID": df["customerID"],
        "ActualChurn": df["Churn"],
        "PredictedChurn": [
            "Yes" if prediction == 1 else "No"
            for prediction in predictions
        ],
        "ChurnProbability": probabilities.round(4)
    })

    output_path = BASE_DIR / "outputs" / "churn_predictions.csv"

    results.to_csv(output_path, index=False)

    print("\nCUSTOMER CHURN PREDICTION")
    print("=" * 35)

    print(f"Customers analyzed: {len(results)}")

    print(
        "Customers predicted to churn:",
        (predictions == 1).sum()
    )

    print(
        "Average predicted churn probability:",
        f"{probabilities.mean():.2%}"
    )

    print("\nSample predictions:")
    print(results.head(10).to_string(index=False))

    print(f"\nPredictions saved to: {output_path}")


if __name__ == "__main__":
    main()