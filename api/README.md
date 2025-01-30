pip install fastapi uvicorn mlflow pandas scikit-learn numpy


uvicorn app:app --host 0.0.0.0 --port 8000 --reload


curl -X 'POST' 'http://127.0.0.1:8000/predict' \
-H 'Content-Type: application/json' \
-d '{
    "Year_of_Birth": 1995,
    "Gender": 1,
    "Origin_Id": 1,
    "Destination_Id": 5,
    "Start_Hour": 14,
    "Start_DayOfWeek": 2
}'
