### **🚲 Mi Bici Trip Duration Prediction Project**
This project focuses on **predicting trip durations for the "Mi Bici" public bike-sharing system** in **Guadalajara, Mexico**. We leverage **machine learning models** to analyze ride patterns and provide real-time trip duration predictions.

🔗 **Data Source:** [Mi Bici Open Data](https://www.mibici.net/es/datos-abiertos/)  
📅 **Training Data:** 2024 trip records  
📅 **Testing Data:** 2025 trip records  

---

## **📌 Project Overview**
This project integrates:
- **MLflow** for experiment tracking & model versioning.
- **FastAPI** for real-time trip duration predictions.
- **Automated retraining** when model performance degrades.

## **📂 Project Structure**
```
.
├───api
│   ├───logs                  # Stores logs from API requests
│   ├───saved_inputs          # Stores API input data for retraining
│   │
│   │   app.py                # FastAPI application
│   │   README.md             # API documentation
│   │   ss_api_test.JPG       # Screenshot of Postman test
│
├───configs
│   │   model1.yaml           # Configuration file for model settings
│
├───data
│   ├───external              # External data sources (if applicable)
│   ├───interim               # Intermediate processed data
│   ├───processed
│   │   ├───2024              # Processed dataset for training
│   │   │   combined_2024.csv  # Combined processed data
│   │   │   test.csv          # Test dataset
│   │   │   train.csv         # Training dataset
│   │
│   └───raw
│       ├───2024              # Raw dataset from Mi Bici 2024
│       │   datos_abiertos_2024_01.csv
│       │   datos_abiertos_2024_02.csv
│       │   datos_abiertos_2024_03.csv
│       │   ...
│       │   datos_abiertos_2024_12.csv
│       │
│       │   nomenclatura_2024_12.csv  # Data dictionary
│
├───docs                      # Project documentation
│
├───MLFlow
│   ├───mlartifacts           # Stores MLflow artifacts
│   ├───mlruns                # MLflow experiment tracking
│   │
│   │   infer.py              # Model inference script
│   │   README.md             # MLflow documentation
│   │   requirements.txt      # Dependencies for MLflow
│   │   retrain_model.py      # Automated retraining script
│   │   ss_mlfow_dashboard.JPG # Screenshot of MLflow dashboard
│   │   train.py              # Model training script
│
├───models
│   │   lr_mae-286.9239_2025-01-29.pkl  # Latest linear regression model
│   │   random_forest_mae-273.4929_2025-01-29.pkl  # Latest random forest model
│
├───notebooks
│   │   1.IRA_data_preprocessing.ipynb   # Data cleaning notebook
│   │   2.IRA_data_vizualization.ipynb   # Exploratory Data Analysis (EDA)
│   │   3.IRA_modeling.ipynb             # Model training and evaluation
│   │   4.IRA_retraining.ipynb           # Model retraining analysis
│
├───references                 # References and additional documentation
│
├───reports
│   ├───figures
│   │   umap_2d_K3.png        # Visualization of UMAP embeddings
│
└───src
    ├───data
    │   │   build_features.py # Feature engineering
    │   │   cleaning.py       # Data cleaning functions
    │   │   ingestion.py      # Data loading
    │   │   labeling.py       # Labeling for supervised learning
    │   │   splitting.py      # Train-test split
    │   │   validation.py     # Data validation
    │
    ├───models
    │   ├───model1
    │   │   │   dataloader.py  # Loads data for training
    │   │   │   hyperparameters_tuning.py # Hyperparameter tuning
    │   │   │   model.py       # Model definition
    │   │   │   predict.py     # Inference script
    │   │   │   preprocessing.py # Preprocessing functions
    │   │   │   train.py       # Training script
    │
    ├───visualization
    │   │   evaluation.py     # Model evaluation scripts
    │   │   exploration.py    # Data exploration and visualization
    │
    │   __init__.py

.env  
.gitignore  
LICENSE  
Makefile  
README.md  
requirements.txt  
```

---


### **1️⃣ Model Training & Logging (MLflow)**
- Uses **Linear Regression** to predict trip durations.
- Trained on **2024 Mi Bici data**.
- Logs:
  - **Hyperparameters**
  - **Performance metrics (MAE, RMSE)**
  - **Model artifacts & schema** in MLflow.

### **2️⃣ Real-time Predictions (FastAPI)**
- Deploys the trained model as a **REST API**.
- **Logs every request & prediction** for monitoring.
- Saves input data (`saved_inputs/prediction_inputs.csv`) for future retraining.

### **3️⃣ Automated Model Retraining**
- **Monitors model performance** on **2025 Mi Bici test data**.
- Retrains the model **if MAE increases beyond a threshold**.
- Uses **both stored API inputs & new Mi Bici data** for retraining.
- Registers **new models in MLflow** if performance improves.

---

## **🛠 Technologies Used**
- **Python** (ML model + API)
- **scikit-learn** (Linear Regression + preprocessing)
- **FastAPI** (Real-time model inference)
- **MLflow** (Model tracking & versioning)
- **pandas & NumPy** (Data processing)
- **joblib** (Model serialization)
- **Logging & CSV storage** (For monitoring & retraining)

---

## **🚀 How It Works**
1️⃣ **Train the Model** (`train.py`):  
   - Loads **Mi Bici 2024 dataset** → Preprocesses → Trains model → Logs in MLflow.

2️⃣ **Deploy API** (`api.py`):  
   - Loads MLflow model → Serves predictions via FastAPI.

3️⃣ **Log Predictions for Retraining**:  
   - API logs input requests → Saves them for future training.

4️⃣ **Monitor & Retrain** (`retrain_model.py`):  
   - Uses **Mi Bici 2025 test data** to check performance.
   - If MAE degrades, merges **new inputs** + **original data**.
   - Retrains & registers **new model in MLflow**.

---

## **📈 Data Overview**
### **Mi Bici Dataset (2024-2025)**
| Column Name      | Description                                        |
|-----------------|----------------------------------------------------|
| Trip_Id        | Unique trip identifier                              |
| User_Id        | Unique user identifier                              |
| Gender         | Gender of the user                                  |
| Year_of_Birth  | Year of birth of the user                          |
| Trip_Start     | Start timestamp of the trip                        |
| Trip_End       | End timestamp of the trip                          |
| Origin_Id      | Origin bike station ID                             |
| Destination_Id | Destination bike station ID                        |
| Trip_Duration  | Total trip duration in seconds                     |
| Start_Hour     | Hour of day when the trip started                  |
| Start_DayOfWeek | Day of the week (Monday=0, Sunday=6)              |

---

## **📌 Running the Project**
### **1️⃣ Install Dependencies**
```bash
pip install -r requirements.txt
```

### **2️⃣ Train & Log the Model**
```bash
python scripts/train.py
```
This will train the model using **Mi Bici 2024 data** and log it in MLflow.

### **3️⃣ Run the FastAPI Server**
```bash
uvicorn scripts.api:app --host 0.0.0.0 --port 8000 --reload
```
The API will be available at **`http://127.0.0.1:8000`**.

### **4️⃣ Make a Prediction**
#### **Using PowerShell**
```powershell
$headers = @{
    "Content-Type" = "application/json"
}

$body = @{
    "Year_of_Birth" = 1995
    "Gender" = 1
    "Origin_Id" = 10
    "Destination_Id" = 50
    "Start_Hour" = 14
    "Start_DayOfWeek" = 2
} | ConvertTo-Json -Depth 10

$response = Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict" -Method Post -Headers $headers -Body $body
Write-Output $response
```

#### **Using Curl**
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"Year_of_Birth": 1995, "Gender": 1, "Origin_Id": 10, "Destination_Id": 50, "Start_Hour": 14, "Start_DayOfWeek": 2}'
```

### **5️⃣ Monitor MLflow**
```bash
mlflow ui
```
Check **experiment logs & model versions** at **`http://127.0.0.1:5000`**.

![MLflow Dashboard](https://raw.githubusercontent.com/Ivanrs297/mlops-end-to-end-mibici/refs/heads/main/MLFlow/ss_mlfow_dashboard.JPG)


### **6️⃣ Schedule Automated Retraining**
Schedule `retrain_model.py` to **run daily** (or as needed) using:
- **Linux (Cron Job)**
  ```bash
  crontab -e
  ```
  Add:
  ```bash
  0 */12 * * * /usr/bin/python3 /path/to/retrain_model.py
  ```
- **Windows (Task Scheduler)**
  - Set **Trigger:** Run **every 12 hours**.
  - Set **Action:** Run:
    ```powershell
    python C:\path\to\retrain_model.py
    ```