# Streaming Service Churn Prediction — Control Center container
# Serves the advanced product API + Retention Console frontend (api.py),
# which supersedes Week3_ML_Model_Development/serving_api.py for
# presentation purposes. See ADVANCED_VERSION.md for why.
#
# Build:  docker build -t churn-console .
# Run:    docker run -p 8000:8000 churn-console
# Open:   http://localhost:8000/        (control center)
#         http://localhost:8000/docs    (Swagger API docs)

FROM python:3.11-slim

WORKDIR /app

# System deps for graphviz-based diagrams (optional, harmless if unused at runtime)
RUN apt-get update && apt-get install -y --no-install-recommends graphviz \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-advanced.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-advanced.txt

COPY . .

# Generate the dataset and train both model versions inside the image, so
# the container is self-contained and doesn't depend on artifacts that
# happen to already be on disk when the image is built.
RUN python Week1_Project_Planning/generate_dataset.py \
    && python Week3_ML_Model_Development/model_training.py \
    && python Week3_ML_Model_Development/model_training_v2.py

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
