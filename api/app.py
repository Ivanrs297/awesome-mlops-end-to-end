import mlflow
import mlflow.pyfunc
import pandas as pd
import logging
from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime
import os

# Set up logging
logging.basicConfig(
    filename="logs/predictions.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Set MLflow tracking server URL
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000/" 
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Load model from MLflow
# MODEL_URI = 'runs:/a89749f054504175abf7bd55ea3c77f3/linear_regression_model'
MODEL_URI = 'models:/linear_regression_model/latest'

model = mlflow.pyfunc.load_model(MODEL_URI)

# Initialize FastAPI app
app = FastAPI()

# Ensure input storage directory exists
os.makedirs("saved_inputs", exist_ok=True)
input_data_file = "saved_inputs/prediction_inputs.csv"

# Define input data structure
class TripFeatures(BaseModel):
    Year_of_Birth: int
    Gender: int
    Origin_Id: int
    Destination_Id: int
    Start_Hour: int
    Start_DayOfWeek: int

@app.get("/")
def home():
    return {"message": "Trip Duration Prediction API is running!"}

@app.post("/predict")
def predict(features: TripFeatures):
    # Convert input data to DataFrame
    input_data = pd.DataFrame([features.dict()])

    # ✅ Ensure data types match MLflow schema expectations
    input_data = input_data.astype({
        "Year_of_Birth": float,  # Convert int to float
        "Gender": int,  # Ensure int
        "Origin_Id": int,  # Ensure int
        "Destination_Id": int,  # Ensure int
        "Start_Hour": int,  # Ensure int
        "Start_DayOfWeek": int  # Ensure int
    })

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Add timestamp and prediction to input data
    input_data["Prediction"] = prediction
    input_data["Timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Save input data for future retraining
    if not os.path.exists(input_data_file):
        input_data.to_csv(input_data_file, index=False)
    else:
        input_data.to_csv(input_data_file, mode='a', header=False, index=False)

    # Log prediction request
    logging.info(f"Prediction made - Input: {features.dict()} - Output: {prediction}")

    return {"Trip_Duration_Prediction": prediction}
