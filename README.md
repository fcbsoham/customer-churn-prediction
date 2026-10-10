# Customer Churn Prediction

A machine learning project that analyzes customer data to predict customer churn and identify customers who may be at risk of leaving a service.

## Project Overview

Customer churn is a major challenge for subscription-based businesses. This project uses machine learning to analyze customer demographics, account information, and service usage to predict the likelihood of customer churn.

The project includes model training, customer-level churn predictions, exploratory analysis visualizations, and an interactive Streamlit dashboard.

## Features

* Data cleaning and preprocessing
* Feature engineering
* Logistic Regression and Random Forest model training
* Model evaluation using accuracy, ROC-AUC, and classification reports
* Customer churn probability predictions
* Interactive Streamlit dashboard
* Batch predictions using uploaded CSV files
* Visualizations of churn distribution, confusion matrix, ROC curve, and model comparison

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Joblib
* Streamlit

## Dataset

This project uses the IBM Telco Customer Churn dataset, containing 7,043 customer records and 21 columns.

The dataset includes customer demographics, subscribed services, contract details, payment methods, tenure, monthly charges, total charges, and churn status.

Download the dataset from the [IBM Telco Customer Churn repository](https://github.com/IBM/telco-customer-churn-on-icp4d).

Place the CSV file in the following location:

`data/WA_Fn-UseC_-Telco-Customer-Churn.csv`

## Model Performance

The following results were obtained during model evaluation:

| Model               | Accuracy | ROC-AUC |
| ------------------- | -------: | ------: |
| Logistic Regression |   73.81% |  0.8413 |
| Random Forest       |   76.22% |  0.8346 |

Logistic Regression was selected based on its higher ROC-AUC score.

These results are from the current evaluation run and may vary if the data split or model configuration changes.

## Project Structure

```text
customer-churn-prediction/
├── app.py
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── outputs/
│   ├── customer_churn_model.joblib
│   ├── churn_predictions.csv
│   ├── churn_analysis_report.txt
│   ├── churn_distribution.png
│   ├── confusion_matrix.png
│   ├── model_comparison.png
│   └── roc_curve.png
├── src/
│   ├── train.py
│   └── predict.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd customer-churn-prediction
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Add the dataset

Download the dataset and place the CSV file in the `data/` directory using the filename specified above.

### 5. Train the models

```bash
python src/train.py
```

### 6. Generate customer predictions

```bash
python src/predict.py
```

### 7. Launch the dashboard

```bash
python -m streamlit run app.py
```

Open the local URL displayed in the terminal to access the dashboard.

## Output Files

The project generates:

* A trained machine learning model
* Customer-level churn predictions
* A churn analysis report
* Churn distribution visualization
* Confusion matrix
* ROC curve
* Model comparison chart

## Future Improvements

* Improve recall and precision for customers likely to churn.
* Perform cross-validation and hyperparameter tuning.
* Add explainability using SHAP or feature importance analysis.
* Develop targeted customer retention recommendations.
* Deploy the dashboard to a cloud platform.

## Disclaimer

Predictions are estimates based on historical customer data. They should support, rather than replace, business decisions.

## Author

Developed as a machine learning and data analytics portfolio project.
