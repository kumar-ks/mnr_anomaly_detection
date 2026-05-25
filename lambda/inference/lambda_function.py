import boto3
import os
import json
import datetime

def lambda_handler(event, context):
    """
    Lambda function that processes daily data through the SageMaker endpoint
    """
    # Create clients
    s3 = boto3.client('s3')
    sagemaker_runtime = boto3.client('sagemaker-runtime')
    sagemaker = boto3.client('sagemaker')
    
    # Get environment variables
    endpoint_name = os.environ['INFERENCE_ENDPOINT_NAME']
    input_bucket = os.environ['INPUT_DATA_BUCKET']
    input_prefix = os.environ['INPUT_DATA_PREFIX']
    output_bucket = os.environ['OUTPUT_DATA_BUCKET']
    output_prefix = os.environ['OUTPUT_DATA_PREFIX']
    
    # Generate today's date for file naming
    today = datetime.datetime.now().strftime('%Y-%m-%d')
    
    try:
        # List objects in the input prefix to find today's file
        response = s3.list_objects_v2(
            Bucket=input_bucket,
            Prefix=f"{input_prefix}daily/"
        )
        print(f"{input_prefix}daily/data-{today}.csv")
        if 'Contents' not in response or len(response['Contents']) == 0:
            print(f"No input file found for date {today}")
            return {
                'statusCode': 404,
                'body': f"No input file found for date {today}"
            }
        
        # Get the latest file
        latest_file = sorted(response['Contents'], key=lambda x: x['LastModified'], reverse=True)[0]
        input_key = latest_file['Key']
        file_size = latest_file['Size']
        
        print(f"Found input file: {input_key} (Size: {file_size} bytes)")
        
        if file_size == 0:
            print("Input file is empty, skipping inference")
            return {
                'statusCode': 400,
                'body': f"Input file {input_key} is empty"
            }
        
        # Download the input CSV from S3
        # print(f"Downloading file from S3: s3://{input_bucket}/{input_key}")
        response = s3.get_object(Bucket=input_bucket, Key=input_key)
        data = response['Body'].read()
        
        # Check if endpoint exists and is ready
        try:
            endpoint_response = sagemaker.describe_endpoint(EndpointName=endpoint_name)
            endpoint_status = endpoint_response['EndpointStatus']
            
            if endpoint_status != 'InService':
                print(f"Endpoint {endpoint_name} is not ready (status: {endpoint_status})")
                return {
                    'statusCode': 503,
                    'body': f"Endpoint {endpoint_name} is not ready (status: {endpoint_status})"
                }
                
            print(f"Endpoint {endpoint_name} is ready with status: {endpoint_status}")
                
        except sagemaker.exceptions.ClientError as e:
            print(f"Error checking endpoint: {str(e)}")
            return {
                'statusCode': 404,
                'body': f"Endpoint {endpoint_name} not found: {str(e)}"
            }
        
        # Invoke the SageMaker endpoint
        print(f"Invoking endpoint {endpoint_name} with data from {input_key}")
        start_time = datetime.datetime.now()
        print(data)
        try:
            response = sagemaker_runtime.invoke_endpoint(
                EndpointName=endpoint_name,
                ContentType='text/csv',
                Body=data
            )
            
            processing_time = (datetime.datetime.now() - start_time).total_seconds()
            print(f"Endpoint invocation successful. Processing time: {processing_time:.2f} seconds")
            
        except Exception as e:
            print(f"Error invoking endpoint: {str(e)}")
            return {
                'statusCode': 500,
                'body': f"Error invoking endpoint: {str(e)}"
            }
        
        # Get results from the endpoint
        results = response['Body'].read()
        
        # Generate output file name
        output_key = f"{output_prefix}/anomaly-results-{today}.csv"
        
        # Upload results to S3
        print(f"Uploading results to S3: s3://{output_bucket}/{output_key}")
        s3.put_object(
            Body=results,
            Bucket=output_bucket,
            Key=output_key,
            ContentType='text/csv'
        )
        
        print(f"Successfully processed data and saved results to s3://{output_bucket}/{output_key}")
        
        return {
            'statusCode': 200,
            'body': {
                'message': 'Inference completed successfully',
                'inputFile': f"s3://{input_bucket}/{input_key}",
                'resultsLocation': f"s3://{output_bucket}/{output_key}",
                'processingTimeSeconds': processing_time
            }
        }
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return {
            'statusCode': 500,
            'body': {
                'message': f'Error during inference: {str(e)}'
            }
        }