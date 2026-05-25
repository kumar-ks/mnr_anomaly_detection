FROM python:3.10-slim

# Set working directory
WORKDIR /opt/ml/code

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the training code
COPY ./train.py .

# Make directory for input data and model output
RUN mkdir -p /opt/ml/input/data/training /opt/ml/model

# Set environment variables
ENV SM_CHANNEL_TRAINING=/opt/ml/input/data/training
ENV SM_MODEL_DIR=/opt/ml/model

# Set up the entry point
ENTRYPOINT ["python", "train.py"]