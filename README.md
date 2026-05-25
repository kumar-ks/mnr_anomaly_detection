# Innovation.MR.Anomaly

System Overview
The MNR Anomaly Detection System is an automated machine learning pipeline that identifies anomalous maintenance and repair (M&R) work orders. The system leverages AWS services to provide a fully automated, serverless solution for model training and inference.

Architecture
The solution implements a cloud-native architecture with these key components:

Docker Containers

Training container with model training code

Inference container with real-time prediction capabilities

AWS Services

Amazon SageMaker for ML infrastructure

AWS Lambda for serverless compute

Amazon S3 for data storage

Amazon EventBridge for orchestration

Amazon ECR for container registry

Automation Components

Scheduled training pipeline (monthly)

Automated model deployment

Daily inference processing


System Components
Training Pipeline
The training pipeline executes monthly to ensure the anomaly detection model remains current with the latest maintenance patterns:

Data Source: Historical work order data from S3

Triggering Mechanism: EventBridge scheduled rule (monthly)

Execution: Lambda function initiates SageMaker training job

Output: ML model artifacts stored in S3

Model Deployment
Once training completes, the system automatically deploys the new model:

Model Registry: SageMaker models catalog

Endpoint Management: Creates endpoint configuration with new model

Deployment Strategy: Blue/green deployment via endpoint update

Inference Pipeline
The daily inference pipeline processes new work orders to identify potential anomalies:

Data Source: Daily work order data in S3

Processing: SageMaker endpoint for real-time inference

Results Storage: Anomaly detection results in S3

Workflow
Monthly Training Cycle:

EventBridge rule triggers Lambda on the 1st day of each month

Lambda initiates SageMaker training job with current data

If training exceeds Lambda timeout, a monitoring rule continues the process

When training completes, the model is registered and deployed

Daily Inference Process:

EventBridge rule triggers daily inference Lambda

Lambda reads new data from S3

Data is sent to SageMaker endpoint for anomaly detection

Results are saved to S3 for reporting and analysis

Configuration
Environment Variables
The system uses environment variables for flexible configuration:

Training Configuration:

TRAINING_IMAGE_URI: xxxx.dkr.ecr.us-east-2.amazonaws.com/innovation.mr.anomaly.training.ecr:latest

INFERENCE_IMAGE_URI: xxxxx.dkr.ecr.us-east-2.amazonaws.com/innovation.mr.anomaly.inference.ecr:latest

SAGEMAKER_ROLE_ARN: arn:aws:iam::xxxx:role/Innovation.MR.Anomaly.Sagemaker.Role

TRAINING_DATA_S3_URI: s3://innovation.mr.anomaly/data/input/monthly/

MODEL_OUTPUT_S3_URI: s3://innovation.mr.anomaly/models/

ENDPOINT_NAME: Innovation-MR-Anomaly-Endpoint

TRAINING_INSTANCE_TYPE: ml.m5.large

INFERENCE_INSTANCE_TYPE: ml.m5.large

AWS Resource Configuration
SageMaker:

Training job configured with 1GB volume size

Maximum runtime of 2 hours per training job

Single instance training configuration

Lambda:

15-minute timeout for training orchestration

Automatic timeout handling for long-running jobs

Monitoring & Resilience
The system implements robust error handling and monitoring:

Training Job Monitoring:

Status polling with graceful timeout handling

Automatic continuation via EventBridge rules for long-running jobs

Endpoint Health Checks:

Verification of endpoint status before inference

Comprehensive error handling and reporting

Logging:

Detailed CloudWatch logs for each component

Status tracking throughout the pipeline

Security
The system follows AWS security best practices:

IAM Roles: Least-privilege permissions for each service

Data Protection: S3 for secure data storage

Network Security: VPC configuration for SageMaker endpoints (optional)

Extensibility
The architecture supports future enhancements:

Multi-Model Support: System design allows for multiple model variants

A/B Testing: Endpoint configuration supports production variants

Scalability: Resource configurations can be adjusted for workload

Maintenance
Regular maintenance activities include:

Monitoring S3 Storage: Review and archive older model artifacts

Review CloudWatch Logs: Monitor for errors or performance issues

Update ML Dependencies: Periodically update container images with latest dependencies
