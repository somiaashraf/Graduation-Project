# MLFLOW PROJECT
# Get conda channels
conda config --show channels

# Build a MLFlow project, if you use one entry point with name (main)
mlflow run . --experiment-name <exp-name> # here it is {readmission-prediction}

# If you have multiple entry points
mlflow run -e random_forest . --experiment-name readmission-prediction
mlflow run -e logistic_regression . --experiment-name readmission-prediction
mlflow run -e xgboost . --experiment-name readmission-prediction
mlflow run -e gradient_boosting . --experiment-name readmission-prediction
mlflow run -e decision_tree . --experiment-name readmission-prediction
mlflow run -e knn . --experiment-name readmission-prediction # Adding KNN entry point



```

```
## MLFLOW Models
``` bash
# serve the model via REST
mlflow models serve -m "path" --port 8000 --env-manager=local
mlflow models serve -m "file:///C:/Users/LAPTOP/Desktop/mlflow-main/mlflow-main/MLFlow%20Project/mlruns/551186705439064227/b0046f6bdd92495d8bbbaa3b31c12750/artifacts/models/GradientBoosting_smotetomek" --port 8001 --env-manager=local

# it will open in this link
http://localhost:8000/invocations
```

``` python
# exmaple of data to be sent


## multiple samples
{
    "dataframe_split": {
         "columns": [
            "gender",
            "age",
            "num_procedures",
            "num_medications",
            "number_emergency",
            "number_inpatient",
            "max_glu_serum",
            "A1Cresult",
            "insulin",
            "change",
            "binary_diabetesMed",
            "readmitted",
            "time_diagnoses_interaction",
            "diag_1_category",
            "diag_2_category",
            "diag_3_category",
            "sulfonylureas",
            "biguanides",
            "thiazolidinediones",
            "meglitinides",
            "alpha_glucosidase_inhibitors",
            "other_combination_therapies"
        ],
        "data": [
            [0, 70.0, 6, 0.2125, 0, 0, 2, 2, 1, 0, 1, 0, 10, 3, 1, 3, 0, 0, 0, 1, 1, 0],
            [0, 80.0, 2, 0.225, 0, 2, 2, 2, 1, 1, 0, 0, 36, 4, 0, 6, 0, 0, 0, 0, 0, 0],
            [0, 70.0, 0, 0.1875, 0, 1, 2, 3, 0, 0, 1, 0, 45, 3, 3, 3, 1, 0, 0, 0, 0, 0],
            [0, 30.0, 0, 0.1625, 0, 4, 2, 2, 1, 1, 0, 0, 36, 3, 1, 1, 0, 0, 0, 0, 0, 0],
            [1, 80.0, 2, 0.2875, 0, 0, 2, 2, 3, 0, 1, 0, 90, 3, 1, 3, 0, 0, 0, 0, 0, 0],
            [1, 80.0, 2, 0.2, 0, 1, 2, 2, 0, 0, 1, 0, 72, 3, 3, 3, 0, 0, 0, 0, 0, 0],
            [0, 80.0, 1, 0.25, 2, 1, 2, 2, 3, 0, 1, 0, 63, 3, 1, 3, 0, 0, 0, 0, 0, 0]
        ]
    }
}
```

``` bash 
# if you want to use curl

curl -X POST \
  http://localhost:8000/invocations \
  -H 'Content-Type: application/json' \
  -d '{
    "dataframe_split": {
        "columns": [
            "gender",
            "age",
            "num_procedures",
            "num_medications",
            "number_emergency",
            "number_inpatient",
            "max_glu_serum",
            "A1Cresult",
            "insulin",
            "change",
            "binary_diabetesMed",
            "readmitted",
            "time_diagnoses_interaction",
            "diag_1_category",
            "diag_2_category",
            "diag_3_category",
            "sulfonylureas",
            "biguanides",
            "thiazolidinediones",
            "meglitinides",
            "alpha_glucosidase_inhibitors",
            "other_combination_therapies"
        ],
        "data": [
            [0, 70.0, 6, 0.2125, 0, 0, 2, 2, 1, 0, 1, 0, 10, 3, 1, 3, 0, 0, 0, 1, 1, 0],
            [0, 80.0, 2, 0.225, 0, 2, 2, 2, 1, 1, 0, 0, 36, 4, 0, 6, 0, 0, 0, 0, 0, 0],
            [0, 70.0, 0, 0.1875, 0, 1, 2, 3, 0, 0, 1, 0, 45, 3, 3, 3, 1, 0, 0, 0, 0, 0],
            [0, 30.0, 0, 0.1625, 0, 4, 2, 2, 1, 1, 0, 0, 36, 3, 1, 1, 0, 0, 0, 0, 0, 0],
            [1, 80.0, 2, 0.2875, 0, 0, 2, 2, 3, 0, 1, 0, 90, 3, 1, 3, 0, 0, 0, 0, 0, 0],
            [1, 80.0, 2, 0.2, 0, 1, 2, 2, 0, 0, 1, 0, 72, 3, 3, 3, 0, 0, 0, 0, 0, 0],
            [0, 80.0, 1, 0.25, 2, 1, 2, 2, 3, 0, 1, 0, 63, 3, 1, 3, 0, 0, 0, 0, 0, 0]
        ]
    }
}'



# if you want to use Powershell
Invoke-RestMethod -Uri "http://localhost:8000/invocations" -Method Post -Headers @{"Content-Type" = "application/json"} -Body '{
    "dataframe_split": {
        "columns": [
            "gender",
            "age",
            "num_procedures",
            "num_medications",
            "number_emergency",
            "number_inpatient",
            "max_glu_serum",
            "A1Cresult",
            "insulin",
            "change",
            "binary_diabetesMed",
            "readmitted",
            "time_diagnoses_interaction",
            "diag_1_category",
            "diag_2_category",
            "diag_3_category",
            "sulfonylureas",
            "biguanides",
            "thiazolidinediones",
            "meglitinides",
            "alpha_glucosidase_inhibitors",
            "other_combination_therapies"
        ],
        "data": [
            [0, 70.0, 6, 0.2125, 0, 0, 2, 2, 1, 0, 1, 0, 10, 3, 1, 3, 0, 0, 0, 1, 1, 0],
            [0, 80.0, 2, 0.225, 0, 2, 2, 2, 1, 1, 0, 0, 36, 4, 0, 6, 0, 0, 0, 0, 0, 0],
            [0, 70.0, 0, 0.1875, 0, 1, 2, 3, 0, 0, 1, 0, 45, 3, 3, 3, 1, 0, 0, 0, 0, 0],
            [0, 30.0, 0, 0.1625, 0, 4, 2, 2, 1, 1, 0, 0, 36, 3, 1, 1, 0, 0, 0, 0, 0, 0]
        ]
    }
}'


```

```