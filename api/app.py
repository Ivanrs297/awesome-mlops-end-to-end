import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
import pandas as pd
import logging
from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime
import os
import numpy as np
import warnings

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

log_path = os.path.abspath("logs/predictions.log")

# Validate that the log file exists (create if it doesn't)
log_dir = os.path.dirname(log_path)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)  # Create the logs directory if needed

# Set up logging
logging.basicConfig(
    filename="logs/predictions.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Set MLflow tracking server URL
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI","http://127.0.0.1:5000/") 
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Load model from MLflow
MODEL_URI = 'models:/linear_regression_model/latest'


# Initialize the MLflow Client
client = MlflowClient()

# Define the model name
model_name = "linear_regression_model"

# Retrieve the latest model version information
latest_version_info = client.get_latest_versions(name=model_name, stages=["None"])[0]
latest_version_number = latest_version_info.version

# Print the latest version number
print(f"The latest version number of the model '{model_name}' is {latest_version_number}.")


model = mlflow.pyfunc.load_model(MODEL_URI)

# Initialize FastAPI app
app = FastAPI()

# Ensure input storage directory exists
input_data_file = os.path.join(BASE_DIR, "saved_inputs", "prediction_inputs.csv")
os.makedirs(os.path.dirname(input_data_file), exist_ok=True)


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
        "Year_of_Birth": np.float64,
        "Gender": np.int64,
        "Origin_Id": np.int64,
        "Destination_Id": np.int64,
        "Start_Hour": np.int32,
        "Start_DayOfWeek": np.int32
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
