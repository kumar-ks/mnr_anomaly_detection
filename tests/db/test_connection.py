import os
import pyodbc
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

server = os.getenv('SQL_SERVER')
database = os.getenv('SQL_DATABASE')
username = os.getenv('SQL_USERNAME')
password = os.getenv('SQL_PASSWORD')

print("=== Testing Fixed Connection String ===")
print(f"Server: {server}")
print(f"Database: {database}")
print(f"Username: {username}")
print()

# Since sqlcmd works, let's try different pyodbc connection string formats
connection_attempts = [
    {
        "name": "Standard format with quotes",
        "conn_str": f'DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server};DATABASE={database};UID="{username}";PWD="{password}";TrustServerCertificate=yes;'
    },
    {
        "name": "Without quotes",
        "conn_str": f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password};TrustServerCertificate=yes;"
    },
    {
        "name": "With connection timeout",
        "conn_str": f'DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server};DATABASE={database};UID="{username}";PWD="{password}";TrustServerCertificate=yes;Connection Timeout=30;Login Timeout=30;'
    },
    {
        "name": "With explicit port",
        "conn_str": f'DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server},1433;DATABASE={database};UID="{username}";PWD="{password}";TrustServerCertificate=yes;'
    }
]

for attempt in connection_attempts:
    print(f"Trying: {attempt['name']}")
    print(f"Connection string: {attempt['conn_str'].replace(password, '*****')}")
    
    try:
        conn = pyodbc.connect(attempt['conn_str'])
        print("✓ SUCCESS!")
        
        # Test a simple query
        cursor = conn.cursor()
        cursor.execute("SELECT @@VERSION")
        version = cursor.fetchone()[0]
        print(f"SQL Server Version: {version[:80]}...")
        
        # Test database access
        cursor.execute("SELECT DB_NAME()")
        db_name = cursor.fetchone()[0]
        print(f"Connected to database: {db_name}")
        
        cursor.close()
        conn.close()
        print("✓ Connection successful and closed")
        
        # If we get here, update the main script with this working connection string
        print(f"\n🎉 WORKING CONNECTION STRING FOUND!")
        print(f"Use this format in your main script:")
        print(f'connection_string = "{attempt["conn_str"].replace(password, "{password}")}"')
        break
        
    except pyodbc.Error as e:
        print(f"✗ FAILED: {e}")
    print("-" * 60)
else:
    print("\n❌ None of the connection string formats worked")
    print("This suggests there might be an issue with the pyodbc driver or Python environment")