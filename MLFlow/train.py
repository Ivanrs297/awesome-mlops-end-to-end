import mlflow
import mlflow.sklearn
import datetime
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from mlflow.models import infer_signature  # For schema inference


# Generate timestamp
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Define MLflow run name
run_name = f"linear_regression_model_{timestamp}"

# Load dataset
df = pd.read_csv("../data/processed/2024/train.csv", parse_dates=['Trip_Start', 'Trip_End'])

# Sample 10% of the data
df = df.sample(frac=0.2, random_state=42)

# Feature Engineering
df['Trip_Duration'] = (df['Trip_End'] - df['Trip_Start']).dt.total_seconds()
df['Start_Hour'] = df['Trip_Start'].dt.hour
df['Start_DayOfWeek'] = df['Trip_Start'].dt.dayofweek

# Drop rows with missing values
df = df.dropna(subset=['Trip_Duration', 'Year_of_Birth', 'Gender', 'Origin_Id', 'Destination_Id'])

# Define Features and Target
X = df[['Year_of_Birth', 'Gender', 'Origin_Id', 'Destination_Id', 'Start_Hour', 'Start_DayOfWeek']]
y = df['Trip_Duration']

# Define categorical and numerical features
categorical_features = ['Gender']
numerical_features = ['Year_of_Birth', 'Origin_Id', 'Destination_Id', 'Start_Hour', 'Start_DayOfWeek']

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ]
)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define Linear Regression pipeline
lr_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Set MLflow experiment
mlflow.set_experiment("trip_duration_prediction")

with mlflow.start_run(run_name=run_name):
    # Train model
    lr_pipeline.fit(X_train, y_train)
    y_pred_lr = lr_pipeline.predict(X_test)

    # Compute Metrics
    mae_lr = mean_absolute_error(y_test, y_pred_lr)
    rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))

    # Infer input-output schema
    signature = infer_signature(X_train, y_pred_lr)

    # Log hyperparameters and metrics
    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_metric("mae", mae_lr)
    mlflow.log_metric("rmse", rmse_lr)

    # Save model with schema
    mlflow.sklearn.log_model(lr_pipeline,
                             "linear_regression_model",
                             signature=signature,
                             registered_model_name="linear_regression_model")

    print(f"Model trained and logged in MLflow with schema.")
    print(f"MLflow run completed. Check UI with 'mlflow ui'")
    print(f"Run Name: {run_name}")

