import mlflow
import pandas as pd
import json

# 1. مكان الموديل
logged_model = 'runs:/b0046f6bdd92495d8bbbaa3b31c12750/models/GradientBoosting_smotetomek'

# 2. تحميل الموديل
loaded_model = mlflow.pyfunc.load_model(logged_model)

# 3. تجهيز الداتا
data_dict = {
  "dataframe_split": {
    "columns": [
      "gender",
      "age",
      "num_procedures",
      "num_medications",
      "number_emergency",
      "number_inpatient",
      "A1Cresult",
      "insulin",
      "change",
      "time_diagnoses_interaction",
      "diag_1_category",
      "diag_2_category",
      "diag_3_category",
      "sulfonylureas",
      "biguanides"
    ],
    "data": [
      [0, 80.0, 2, 0.3379, 0, 0, 2, 2, 0, 104, 2, 3, 2, 1, 0],
      [0, 90.0, 3, 0.21249, 0, 0, 2, 2, 0, 96, 3, 0, 3, 0, 0],
      [0, 40.0, 2, 0.199, 0, 0, 2, 2, 1, 81, 1, 3, 5, 0, 0],
      [1, 60.0, 0, 0.125, 0, 0, 2, 2, 0, 49, 0, 1, 0, 2, 0],
      [0, 40.0, 0, 0.175, 1, 0, 2, 0, 0, 56, 3, 1, 1, 0, 1],
      [1, 80.0, 1, 0.375, 0, 0, 2, 2, 1, 80, 3, 3, 3, 0, 0],
      [0, 60.0, 5, 0.012499, 0, 0, 2, 2, 1, 8, 3, 5, 4, 0, 0],
      [1, 60.0, 5, 0.15, 0, 0, 2, 3, 0, 108, 5, 3, 5, 0, 0],
      [1, 50.0, 4, 0.199, 0, 0, 2, 2, 0, 32, 3, 3, 3, 1, 0]
    ]
  }
}

# تحويلها الى DataFrame
columns = data_dict["dataframe_split"]["columns"]
data = data_dict["dataframe_split"]["data"]
df = pd.DataFrame(data, columns=columns)

# 4. عمل التوقع
predictions = loaded_model.predict(df)

# 5. عرض النتائج
print(predictions)
