pip install mlflow pandas scikit-learn numpy joblib


mlflow ui


Since the new predictions do not include Trip_Start and Trip_End, we can:

Reconstruct Trip_Start using:
Start_Hour
Start_DayOfWeek
Assume the current year.
Calculate Trip_End as:
Trip_Start + Trip_Duration_Prediction (in seconds).