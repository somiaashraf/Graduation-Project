import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse, os, warnings

import mlflow
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, f1_score
from imblearn.combine import SMOTETomek

warnings.filterwarnings("ignore")

## --------------------- Data Preparation ---------------------------- ##
def load_and_prepare_data(filepath):
    df = pd.read_csv(filepath)
    
    cols_to_drop = [
        'other_combination_therapies', 
        'alpha_glucosidase_inhibitors',
        'meglitinides',
        'thiazolidinediones',
        'max_glu_serum',
        'binary_diabetesMed'
    ]
    df.drop(columns=[col for col in cols_to_drop if col in df.columns], inplace=True)

    X = df.drop(columns=['readmitted'])
    y = df['readmitted']

    return X, y


## --------------------- Resampling ---------------------------- ##
def resample_data_before_split(X, y):
    smt = SMOTETomek(random_state=42)
    X_resampled, y_resampled = smt.fit_resample(X, y)
    return X_resampled, y_resampled


## --------------------- Modeling ---------------------------- ##
def train_and_evaluate(X_train, y_train, X_test, y_test, plot_name):
    mlflow.set_experiment("readmission-prediction")
    with mlflow.start_run(run_name="LogisticRegression_Model") as run:
        model_name = f"LogisticRegression_{plot_name}"
        mlflow.set_tags({
            "model_name": model_name,
            "experiment_type": "LogisticRegression",
            "description": "Predicting Readmission with Logistic Regression + SMOTETomek before splitting"
        })

        mlflow.log_params({
            "model": "LogisticRegression",
            "class_weight": "balanced",
            "max_iter": 1000
        })

        model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        report = classification_report(y_test, y_pred)

        mlflow.log_metrics({'accuracy': acc, 'f1_score': f1})
        mlflow.log_text(report, f"{model_name}_classification_report.txt")
        mlflow.sklearn.log_model(model, f"models/{model_name}")

        print(f"Accuracy: {acc}")
        print(f"Classification Report:\n{report}")

        # Confusion Matrix
        plt.figure(figsize=(6, 5))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
        plt.title(f"Confusion Matrix - {plot_name}")
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        mlflow.log_figure(plt.gcf(), f'{model_name}_conf_matrix.png')
        plt.close()

        # ROC Curve
        if len(np.unique(y_test)) == 2 and hasattr(model, "predict_proba"):
            fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
            roc_auc = auc(fpr, tpr)
            plt.figure(figsize=(6, 4))
            plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
            plt.plot([0, 1], [0, 1], linestyle='--')
            plt.xlabel("FPR")
            plt.ylabel("TPR")
            plt.title(f"ROC Curve - {plot_name}")
            plt.legend(loc="lower right")
            mlflow.log_figure(plt.gcf(), f'{model_name}_roc.png')
            plt.close()


## --------------------- Main Function ---------------------------- ##
def main(filepath):
    X, y = load_and_prepare_data(filepath)

    # ✅ SMOTETomek قبل التقسيم
    X_resampled, y_resampled = resample_data_before_split(X, y)

    # التقسيم بعد إعادة التوازن
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.3, random_state=42, stratify=y_resampled
    )

    train_and_evaluate(X_train, y_train, X_test, y_test, plot_name='smotetomek_before_split')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filepath', '-f', type=str, required=True, help="Path to the CSV file")
    args = parser.parse_args()
    main(filepath=args.filepath)
