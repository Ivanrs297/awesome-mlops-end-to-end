import mlflow
import mlflow.sklearn
import pandas as pd

# Load dataset (Replace with actual dataset)
df = pd.read_csv("data/dataset.csv")  # Change file path accordingly

# Prepare test data
X = df.drop(columns=['target'])  # Replace 'target' with actual target column

# Load latest model from MLflow
model_uri = "models:/random_forest_model/latest"
rf_pipeline_loaded = mlflow.sklearn.load_model(model_uri)

# Make predictions
y_pred = rf_pipeline_loaded.predict(X)

print("Predictions:", y_pred[:10])  # Show first 10 predictions
