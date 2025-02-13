# Use Python base image
FROM python:3.10

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r MLFlow/requirements.txt

# Run retraining script on container startup
CMD ["python", "MLFlow/retrain_model.py"]
