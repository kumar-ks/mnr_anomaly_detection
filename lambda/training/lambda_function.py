import boto3
import os
import datetime
import time
import json

def lambda_handler(event, context):
    """
    Lambda function that:
    1. Starts a new SageMaker training job
    2. Waits for it to complete
    3. Creates a new model using the trained artifacts
    4. Updates the endpoint with the new model
    """
    # Initialize SageMaker client
    sagemaker = boto3.client('sagemaker')
    
    # Generate timestamp for unique naming
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    
    # Get environment variables for configuration
    training_image = os.environ['TRAINING_IMAGE_URI']
    sagemaker_role = os.environ['SAGEMAKER_ROLE_ARN']
    training_data_uri = os.environ['TRAINING_DATA_S3_URI']
    model_output_uri = os.environ['MODEL_OUTPUT_S3_URI']
    endpoint_name = os.environ['ENDPOINT_NAME']
    
    # 1. START TRAINING JOB
    print("Starting training job...")
    
    # Generate unique training job name
    training_job_name = f"mnr-anomaly-training-{timestamp}"
    
    # Define training job parameters
    training_params = {
        'TrainingJobName': training_job_name,
        'AlgorithmSpecification': {
            'TrainingImage': training_image,
            'TrainingInputMode': 'File'
        },
        'RoleArn': sagemaker_role,
        'InputDataConfig': [
            {
                'ChannelName': 'training',
                'DataSource': {
                    'S3DataSource': {
                        'S3DataType': 'S3Prefix',
                        'S3Uri': training_data_uri,
                        'S3DataDistributionType': 'FullyReplicated'
                    }
                },
                'ContentType': 'text/csv'
            }
        ],
        'OutputDataConfig': {
            'S3OutputPath': model_output_uri
        },
        'ResourceConfig': {
            'InstanceType': os.environ.get('TRAINING_INSTANCE_TYPE', 'ml.m5.xlarge'),
            'InstanceCount': 1,
            'VolumeSizeInGB': 1
        },
        'StoppingCondition': {
            'MaxRuntimeInSeconds': 7200  # 2 hours max runtime
        }
    }
    
    # Create the training job
    sagemaker.create_training_job(**training_params)
    print(f"Created training job: {training_job_name}")
    
    # 2. WAIT FOR TRAINING JOB TO COMPLETE
    print("Waiting for training job to complete...")
    
    # Get the training job status (initial status will be InProgress)
    status = 'InProgress'
    
    # Check status periodically (every 30 seconds)
    # Lambda has max execution time of 15 mins, so we'll timeout if not done by then
    # For longer jobs, you should use Step Functions instead
    start_time = time.time()
    max_wait_time = 840  # 14 minutes (leaving 1 min buffer for Lambda's 15 min limit)
    
    while status == 'InProgress':
        # Check if we're running out of time
        elapsed_time = time.time() - start_time
        if elapsed_time > max_wait_time:
            print(f"Training job still running after {elapsed_time:.1f} seconds")
            print("Lambda execution time limit approaching - exiting and relying on CloudWatch Events")
            
            # Create CloudWatch Event rule to check job completion
            create_job_completion_rule(training_job_name, endpoint_name)
            
            return {
                'statusCode': 200,
                'body': {
                    'message': f'Training job {training_job_name} still in progress',
                    'status': 'InProgress',
                    'trainingJob': training_job_name
                }
            }
        
        # Get current job status
        response = sagemaker.describe_training_job(TrainingJobName=training_job_name)
        status = response['TrainingJobStatus']
        
        if status == 'Completed':
            print(f"Training job completed after {elapsed_time:.1f} seconds")
            break
        elif status == 'Failed':
            print(f"Training job failed after {elapsed_time:.1f} seconds")
            print(f"Failure reason: {response.get('FailureReason', 'Unknown')}")
            return {
                'statusCode': 500,
                'body': {
                    'message': f'Training job {training_job_name} failed',
                    'reason': response.get('FailureReason', 'Unknown'),
                    'trainingJob': training_job_name
                }
            }
        elif status == 'Stopped':
            print(f"Training job was stopped after {elapsed_time:.1f} seconds")
            return {
                'statusCode': 500,
                'body': {
                    'message': f'Training job {training_job_name} was stopped',
                    'trainingJob': training_job_name
                }
            }
        else:
            # Still in progress - wait for 30 seconds before checking again
            print(f"Training job status: {status} - elapsed time: {elapsed_time:.1f} seconds")
            time.sleep(30)
    
    # 3. CREATE NEW MODEL FROM TRAINING ARTIFACTS
    print("Creating new model from training artifacts...")
    
    # Generate unique model name
    model_name = f"mnr-anomaly-model-{timestamp}"
    
    # Get the S3 path where model artifacts are stored
    model_output_uri = model_output_uri.rstrip('/') 
    model_artifacts = f"{model_output_uri}/{training_job_name}/output/model.tar.gz"
    
    # Get inference image URI
    inference_image = os.environ['INFERENCE_IMAGE_URI']
    
    # Create model
    model_params = {
        'ModelName': model_name,
        'PrimaryContainer': {
            'Image': inference_image,
            'ModelDataUrl': model_artifacts
        },
        'ExecutionRoleArn': sagemaker_role
    }
    
    time.sleep(10)  
    sagemaker.create_model(**model_params)
    print(f"Created model: {model_name}")
    
    # 4. UPDATE ENDPOINT WITH NEW MODEL
    print(f"Updating endpoint: {endpoint_name} with new model: {model_name}")
    
    # First, create a new endpoint configuration
    endpoint_config_name = f"mnr-anomaly-config-{timestamp}"
    
    endpoint_config_params = {
        'EndpointConfigName': endpoint_config_name,
        'ProductionVariants': [
            {
                'VariantName': 'Default',
                'ModelName': model_name,
                'InitialInstanceCount': 1,
                'InstanceType': os.environ.get('INFERENCE_INSTANCE_TYPE', 'ml.m5.large')
            }
        ]
    }
    
    sagemaker.create_endpoint_config(**endpoint_config_params)
    print(f"Created endpoint configuration: {endpoint_config_name}")
    
    # Check if endpoint exists
    try:
        sagemaker.describe_endpoint(EndpointName=endpoint_name)
        endpoint_exists = True
    except sagemaker.exceptions.ClientError:
        endpoint_exists = False
    
    if endpoint_exists:
        # Update existing endpoint
        sagemaker.update_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_config_name
        )
        print(f"Updated existing endpoint: {endpoint_name}")
    else:
        # Create new endpoint
        sagemaker.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_config_name
        )
        print(f"Created new endpoint: {endpoint_name}")
    
    print("Endpoint update initiated - it may take several minutes to complete")
    
    return {
        'statusCode': 200,
        'body': {
            'message': f'Training completed and endpoint {endpoint_name} update initiated',
            'trainingJob': training_job_name,
            'model': model_name,
            'endpointConfig': endpoint_config_name
        }
    }

def create_job_completion_rule(training_job_name, endpoint_name):
    """
    Create a CloudWatch Events rule to monitor training job completion
    and trigger endpoint update when training completes.
    """
    events = boto3.client('events')
    lambda_client = boto3.client('lambda')
    
    # Create rule to monitor training job completion
    rule_name = f"training-completion-{training_job_name}"
    
    # Define the event pattern to match SageMaker training job completion
    event_pattern = {
        "source": ["aws.sagemaker"],
        "detail-type": ["SageMaker Training Job State Change"],
        "detail": {
            "TrainingJobName": [training_job_name],
            "TrainingJobStatus": ["Completed"]
        }
    }
    
    # Create the rule
    events.put_rule(
        Name=rule_name,
        EventPattern=json.dumps(event_pattern),
        State="ENABLED",
        Description=f"Monitor completion of training job {training_job_name}"
    )
    
    # Get this Lambda function's ARN
    lambda_arn = context.invoked_function_arn
    
    # Add permission for CloudWatch Events to invoke the Lambda
    try:
        lambda_client.add_permission(
            FunctionName=lambda_arn,
            StatementId=f"AllowEventRule-{rule_name}",
            Action="lambda:InvokeFunction",
            Principal="events.amazonaws.com",
            SourceArn=f"arn:aws:events:{'us-east-2'}:{os.environ['AWS_ACCOUNT_ID']}:rule/{rule_name}"
        )
    except lambda_client.exceptions.ResourceConflictException:
        # Permission already exists
        pass
    
    # Create input for the continuation Lambda
    input_template = {
        "action": "update_endpoint",
        "trainingJobName": training_job_name,
        "endpointName": endpoint_name
    }
    
    # Add the target
    events.put_targets(
        Rule=rule_name,
        Targets=[
            {
                'Id': '1',
                'Arn': lambda_arn,
                'Input': json.dumps(input_template)
            }
        ]
    )
    
    print(f"Created CloudWatch Events rule {rule_name} to monitor training completion")