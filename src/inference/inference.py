import os
import pandas as pd
import joblib
from flask import Flask, request
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
logger.addHandler(logging.StreamHandler(sys.stdout))

# Initialize Flask app
app = Flask(__name__)

# Path to model artifacts - default to SageMaker paths with fallbacks
model_dir = os.environ.get('SM_MODEL_DIR', os.environ.get('MODEL_DIR', './models'))

# Load model artifacts at startup
try:
    model_path = os.path.join(model_dir, 'model_artifacts.joblib')
    logger.info(f"Loading model from {model_path}")
    model_artifacts = joblib.load(model_path)
    logger.info(f"Model loaded with {len(model_artifacts['groups'])} group configurations")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    model_artifacts = None

@app.route('/ping', methods=['GET'])
def ping():
    """Health check endpoint"""
    if model_artifacts:
        return "OK", 200
    return "Model not loaded", 404

@app.route('/invocations', methods=['POST'])
def invoke():
    """Inference endpoint required by SageMaker"""
    try:
        # Parse input data
        if request.content_type == 'text/csv':
            data = request.data.decode('utf-8')
            logger.info("Received CSV data")
            
            try:
                df = pd.read_csv(pd.io.common.StringIO(data))
                logger.info(f"Parsed DataFrame with shape: {df.shape}")
            except Exception as e:
                logger.error(f"Error parsing CSV: {e}")
                return f"Error parsing CSV: {e}", 400
        else:
            logger.error(f"Unsupported content type: {request.content_type}")
            return "This endpoint only supports CSV input", 415

        # Ensure model is loaded
        if not model_artifacts:
            logger.error("Model not loaded")
            return "Model not loaded", 500

        required_cols = ['repairCode', 'shop_code', 'RepairTypeCode', 'cdx_damage', 'total_repair_hours']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            return f"Missing required columns: {missing_cols}", 400
        
        exclude_shop_codes = ['246', '2KB', '4HT', '4KB', '4NT', '5BF', '5KB', '5PI', '5PJ', '5PK', '5PL', '5PM', '5SE', '5TR', '6UF']
        df = df[~df['shop_code'].isin(exclude_shop_codes)]
        df = df[[
            'workorder_date', 'submit_date', 'days_difference', 'repairCode',
            'vendor_est_no', 'workorder_no', 'shop_code', 'shopName', 'IMRS_Status',
            'workorder_labor', 'workorder_parts', 'RepairTypeCode', 'cdx_damage',
            'cdx_location', 'damage_description', 'total_repair_hours'
        ]]
        

        # Run anomaly detection
        results = []
        min_cluster_size = model_artifacts.get('min_cluster_size', 3)

        # Group data
        try:
            df_combination = df.groupby(['repairCode', 'shop_code', 'RepairTypeCode', 'cdx_damage'])
            logger.info(f"Created {len(df_combination)} groups")
        except Exception as e:
            logger.error(f"Error grouping data: {e}")
            return f"Error grouping data: {e}", 400

        # Process each group
        groups_processed = 0
        anomaly_count = 0
        
        for name, group in df_combination:
            # If model doesn't have this group, mark as non-anomaly
            if str(name) not in model_artifacts['groups']:
                group['anomaly'] = False
                results.append(group)
                continue

            # Get model artifacts for this group
            group_artifacts = model_artifacts['groups'][str(name)]
            scaler = group_artifacts['scaler']
            small_clusters = group_artifacts.get('small_clusters', [])
            group_median = group_artifacts.get('median', 0)
            
            # Scale the data
            features = group[['total_repair_hours']]
            scaled_features = scaler.transform(features)

            # Assign clusters and detect anomalies
            try:            
        
                # Run clustering with same parameters as training
                model = AgglomerativeClustering(
                    n_clusters=None, 
                    distance_threshold=model_artifacts.get('distance_threshold', 1.0),
                    linkage='ward'
                )
                group['cluster'] = model.fit_predict(scaled_features)

                # Identify small clusters
                value_counts = group['cluster'].value_counts()
                current_small_clusters = value_counts[value_counts < min_cluster_size].index.tolist()
                
                # Mark points in small clusters as anomalies
                group['anomaly'] = group['cluster'].isin(current_small_clusters)
                
                # Filter low-end false positives
                group.loc[group['total_repair_hours'] <= group_median, 'anomaly'] = False
                                # Count anomalies
                anomaly_count += group['anomaly'].sum()

            except Exception as e:
                logger.error(f"Error processing group {name}: {e}")
                group['anomaly'] = False    

            results.append(group)
        
            groups_processed += 1
            if groups_processed % 100 == 0:
                logger.info(f"Processed {groups_processed} groups")

        # Combine all results
        if results:
            final_df = pd.concat(results)
            logger.info(f"Returning results: {len(final_df)} rows, {anomaly_count} anomalies")
            return final_df.to_csv(index=False), 200, {'Content-Type': 'text/csv'}
        else:
            logger.warning("No results to return")
            return "No valid data to process", 400
            
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return f"Error: {str(e)}", 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    logging.info(f"Starting Flask server on port {port}")
    app.run(host='0.0.0.0', port=port)