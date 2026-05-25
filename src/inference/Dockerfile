FROM python:3.10-slim

# Set working directory
WORKDIR /opt/ml/code

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the inference code
COPY ./inference.py .

# Make directories for model and data
RUN mkdir -p /opt/ml/model /opt/ml/input/data /opt/ml/output/data

# Set environment variables
ENV PORT=8080
ENV MODEL_DIR=/opt/ml/model
ENV DATA_PATH=/opt/ml/input/data
ENV OUTPUT_PATH=/opt/ml/output/data

# Expose port 8080 for SageMaker
EXPOSE 8080

# Set up the entry point
ENTRYPOINT ["python", "inference.py"]