from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import os
import uvicorn
import nest_asyncio
from pyngrok import ngrok

# Set your Ngrok authentication token
ngrok.set_auth_token("2w9xGxAvHLQdIoZSShkjDooZpQz_3RhUfP2TezsP3vehcsdr")  # Replace "YOUR_AUTHTOKEN" with your actual token from ngrok.com

# Load the trained model
model_path =r"C:\Users\LAPTOP\Desktop\mlflow-main\mlflow-main\MLFlow Project\model.pkl"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ Model file not found at {model_path}")

model = joblib.load(model_path)

# Feature columns used during model training
FEATURE_COLUMNS = [
    "gender", "age", "num_procedures", "num_medications", 
    "number_emergency", "number_inpatient", "A1Cresult", 
    "insulin", "change", "time_diagnoses_interaction", 
    "diag_1_category", "diag_2_category", "diag_3_category",
    "sulfonylureas", "biguanides"
]

# Input data schema
class InputData(BaseModel):
    data: list

# Initialize FastAPI app
app = FastAPI(
    title="Health Risk Prediction API",
    description="This API predicts health risks based on patient data using a trained machine learning model.",
    version="1.0.0"
)

# Prediction endpoint
@app.post("/prediction/")
def predict(input_data: dict):
    try:
        # لو البيانات جايه بتنسيق dataframe_split
        if "dataframe_split" in input_data:
            df_dict = input_data["dataframe_split"]
            df = pd.DataFrame(data=df_dict["data"], columns=df_dict["columns"])
        # لو البيانات جايه بالطريقة العادية
        elif "data" in input_data:
            df = pd.DataFrame(input_data["data"], columns=FEATURE_COLUMNS)
        else:
            raise ValueError("Invalid input format. Expected 'data' or 'dataframe_split'.")

        # Make predictions
        predictions = model.predict(df)

        return predictions.tolist()  # لو عايز يطبع فقط النتائج

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"❌ Error: {str(e)}")


# Allow Uvicorn to run inside Jupyter/Colab
nest_asyncio.apply()

# Open ngrok tunnel to expose local server
public_url = ngrok.connect(8000)
print(f"\n🚀 Public URL: {public_url}\n🔗 Localhost: http://127.0.0.1:8000\n")

# Run the FastAPI app
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
