import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

def main():
    logger.info("Starting training job...")
    
    # Get paths from environment variables with SageMaker defaults
    training_dir = os.environ.get('SM_CHANNEL_TRAINING', './data/input')
    model_dir = os.environ.get('SM_MODEL_DIR', './models')
    
    # List files in the training directory
    logger.info(f"Listing training directory contents: {os.listdir(training_dir)}")
    
    # Find CSV files in training directory
    csv_files = [f for f in os.listdir(training_dir) if f.endswith('.csv')]
    if not csv_files:
        raise ValueError(f"No CSV files found in {training_dir}")
    
    # Use the first CSV file (or implement logic to select a specific file)
    input_file = os.path.join(training_dir, csv_files[0])
    logger.info(f"Using input file: {input_file}")
    
    # Load data
    df = pd.read_csv(input_file)
    logger.info(f"Loaded data with shape {df.shape}")
    
    # Data preprocessing
    logger.info("Preprocessing data...")
    try:
        df = df[df['reject_msg_vendor'].isnull()]
        exclude_shop_codes = ['246', '2KB', '4HT', '4KB', '4NT', '5BF', '5KB', '5PI', '5PJ', '5PK', '5PL', '5PM', '5SE', '5TR', '6UF']
        df = df[~df['shop_code'].isin(exclude_shop_codes)]
        df = df[[
            'workorder_date', 'submit_date', 'days_difference', 'repairCode',
            'vendor_est_no', 'workorder_no', 'shop_code', 'shopName', 'IMRS_Status',
            'workorder_labor', 'workorder_parts', 'RepairTypeCode', 'cdx_damage',
            'cdx_location', 'damage_description', 'total_repair_hours'
        ]]
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        # Gracefully handle missing columns
        required_columns = ['repairCode', 'shop_code', 'RepairTypeCode', 'cdx_damage', 'total_repair_hours']
        if not all(col in df.columns for col in required_columns):
            missing_cols = [col for col in required_columns if col not in df.columns]
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    logger.info("Creating grouped data and fitting scalers...")
    
    # Group data by required fields
    try:
        df_combination = df.groupby(['repairCode', 'shop_code', 'RepairTypeCode', 'cdx_damage'])
    except Exception as e:
        logger.error(f"Error during grouping: {e}")
        raise

    # Dictionary to store model parameters
    model_artifacts = {
        'distance_threshold': 1.0,
        'min_cluster_size': 3,  # Minimum cluster size to not be considered anomaly
        'min_rows_required': 30,
        'groups': {}
    }
    
    # Process each group
    groups_processed = 0
    for name, group in df_combination:
        if len(group) < model_artifacts['min_rows_required']:
            continue
            
        # Extract and scale the target feature
        features = group[['total_repair_hours']]
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        
        # Run Agglomerative Clustering with distance threshold
        model = AgglomerativeClustering(n_clusters=None, distance_threshold=model_artifacts['distance_threshold'])
        cluster_labels = model.fit_predict(scaled_features)
        group['cluster'] = cluster_labels
        
        # Identify small clusters as potential anomalies
        value_counts = group['cluster'].value_counts()
        small_clusters = value_counts[value_counts < model_artifacts['min_cluster_size']].index.tolist()
        
        # Calculate cluster centroids
        cluster_centroids = group.groupby('cluster')['total_repair_hours'].mean().to_dict()
        
        # Get group median for filtering low-end false positives
        group_median = group['total_repair_hours'].median()
        
        # Store the trained artifacts
        model_artifacts['groups'][str(name)] = {
            'scaler': scaler,
            'centroids': cluster_centroids,
            'median': group_median,
            'small_clusters': small_clusters
        }
        
    logger.info(f"Processing Completed.")
    
    # Save model artifacts
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'model_artifacts.joblib')
    joblib.dump(model_artifacts, model_path)
    logger.info(f"Model artifacts saved to {model_path}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise