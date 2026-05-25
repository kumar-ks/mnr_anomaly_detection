import os
import pyodbc
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

server = os.getenv('SQL_SERVER')
database = os.getenv('SQL_DATABASE')
username = os.getenv('SQL_USERNAME')
password = os.getenv('SQL_PASSWORD')

class AnomalyDetectionPipeline:
    def __init__(self):
        self.connection_string = (
            f"DRIVER={{ODBC Driver 17 for SQL Server}};"
            f"SERVER={server};"
            f"DATABASE={database};"
            f"UID={username};"
            f"PWD={password};"
            f"TrustServerCertificate=yes;"
        )
        self.conn = None
    
    def connect(self):
        """Establish database connection"""
        try:
            self.conn = pyodbc.connect(self.connection_string)
            print("✓ Database connection successful!")
            return True
        except pyodbc.Error as e:
            print(f"✗ Connection failed: {e}")
            return False
    
    def fetch_data(self, sql_query):
        """Fetch data from SQL Server using provided query"""
        try:
            df = pd.read_sql(sql_query, self.conn)
            print(f"✓ Data fetched successfully: {len(df)} rows, {len(df.columns)} columns")
            return df
        except Exception as e:
            print(f"✗ Error fetching data: {e}")
            return None
    
    def apply_ml_logic(self, df):
        """Apply ML logic - currently adds is_anomaly column with False values"""
        try:
            # Add anomaly detection column (placeholder for ML logic)
            df['is_anomaly'] = False
            
            # You can replace this with actual ML model logic later
            # Example: df['is_anomaly'] = your_ml_model.predict(df)
            
            print("✓ ML logic applied - added 'is_anomaly' column")
            return df
        except Exception as e:
            print(f"✗ Error applying ML logic: {e}")
            return None
    
    def table_exists(self, table_name, schema="dbo", database='IMRS'):
        """Check if table exists in database"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT COUNT(*) 
                FROM INFORMATION_SCHEMA.TABLES 
                WHERE TABLE_CATALOG = ? AND TABLE_SCHEMA = ? AND TABLE_NAME = ?
            """, (database, schema, table_name))
            exists = cursor.fetchone()[0] > 0
            cursor.close()
            return exists
        except Exception as e:
            print(f"✗ Error checking table existence: {e}")
            return False
    
    def create_table_from_dataframe(self, df, table_name, schema="dbo", database='IMRS'):
        """Create table based on DataFrame structure"""
        try:
            cursor = self.conn.cursor()
            
            # Generate CREATE TABLE statement based on DataFrame dtypes
            columns = []
            for col, dtype in df.dtypes.items():
                if dtype == 'object':
                    columns.append(f"[{col}] NVARCHAR(MAX)")
                elif dtype == 'int64':
                    columns.append(f"[{col}] BIGINT")
                elif dtype == 'float64':
                    columns.append(f"[{col}] FLOAT")
                elif dtype == 'bool':
                    columns.append(f"[{col}] BIT")
                elif dtype == 'datetime64[ns]':
                    columns.append(f"[{col}] DATETIME2")
                else:
                    columns.append(f"[{col}] NVARCHAR(MAX)")
            
            columns_sql = ", ".join(columns)
            create_sql = f"CREATE TABLE [{database}].[{schema}].[{table_name}] ({columns_sql})"
            cursor.execute(create_sql)
            self.conn.commit()
            cursor.close()
            print(f"✓ Table '{table_name}' created successfully")
            return True
        except Exception as e:
            print(f"✗ Error creating table: {e}")
            return False
    
    def upload_data(self, df, table_name, if_exists='append'):
        """Upload DataFrame to SQL Server table"""
        try:
            # Check if table exists
            if not self.table_exists(table_name):
                print(f"Table '{table_name}' doesn't exist. Creating...")
                if not self.create_table_from_dataframe(df, table_name, schema="dbo", database="IMRS"):
                    return False
            
            # Insert data using pandas to_sql with pyodbc connection
            # Create SQLAlchemy engine from pyodbc connection for pandas compatibility
            from sqlalchemy import create_engine
            engine = create_engine(f"mssql+pyodbc://", creator=lambda: self.conn)
            
            df.to_sql(table_name, con=engine, if_exists=if_exists, index=False)
            print(f"✓ Data uploaded successfully to '{table_name}': {len(df)} rows")
            return True
        except Exception as e:
            print(f"✗ Error uploading data: {e}")
            return False
    
    def run_pipeline(self, sql_query, output_table_name, if_exists='append'):
        """Run the complete anomaly detection pipeline"""
        print("Starting Anomaly Detection Pipeline...")
        print("=" * 50)
        
        # Step 1: Connect to database
        if not self.connect():
            return False
        
        # Step 2: Fetch data
        df = self.fetch_data(sql_query)
        if df is None:
            return False
        
        # Step 3: Apply ML logic
        processed_df = self.apply_ml_logic(df)
        if processed_df is None:
            return False
        
        # Step 4: Upload processed data
        success = self.upload_data(processed_df, output_table_name, if_exists)
        
        # Step 5: Close connection
        if self.conn:
            self.conn.close()
            print("✓ Database connection closed")
        
        print("=" * 50)
        if success:
            print("✓ Pipeline completed successfully!")
        else:
            print("✗ Pipeline failed!")
        
        return success

# Example usage
if __name__ == "__main__":
    # Initialize pipeline
    pipeline = AnomalyDetectionPipeline()

    # Read SQL query from file
    sql_file_path = os.path.join(os.path.dirname(__file__), "HaileeSmith.WOs.sql")
    with open(sql_file_path, "r") as f:
        sql_query = f.read()
    
    
    # Define output table name
    output_table = "InnovationTest"
    
    # Run the pipeline
    pipeline.run_pipeline(sql_query, output_table, if_exists='replace')