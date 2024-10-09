

from fastapi import FastAPI, Request, HTTPException
from huggingface_hub import InferenceClient
import os
import sqlite3
import requests  # To make HTTP requests to other endpoints
import logging
import json
import pandas as pd
# import pandasql as psql
from pandasql import sqldf
import threading
from datetime import datetime

# # Initialize logging
logging.basicConfig(level=logging.INFO)

app = FastAPI()


DATABASE_PATH_SQL = 'your_database.db'

csv_file_path = 'Sales_Data.csv'

import numpy as np

def convert_numpy_types(data):
    if isinstance(data, dict):
        return {k: convert_numpy_types(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_numpy_types(v) for v in data]
    elif isinstance(data, np.generic):
        return data.item()
    else:
        return data


def run_sql_on_csv(sql_query: str):
    try:
        sales = pd.read_csv(csv_file_path)  # Adjust the file path as necessary

        with sqlite3.connect(":memory:") as conn:
            # Load the DataFrame into the SQLite database
            sales.to_sql("sales", conn, index=False, if_exists="replace")

            # Execute the SQL query and get the result as a DataFrame
            result_df = pd.read_sql_query(sql_query, conn)

        # Convert the DataFrame to a list of dictionaries
        result = result_df.to_dict(orient='records')

        # Convert NumPy data types to native Python types
        result = convert_numpy_types(result)

        return result

    except Exception as e:
        logging.error(f"Error executing query on CSV: {e}")
        raise HTTPException(status_code=500, detail=f"Error executing query on CSV: {e}")
    


# Function to execute SQL query on the database and return the results
def execute_sql_query(sql_query: str):
    try:
        # Connect to the SQLite database
        conn = sqlite3.connect(DATABASE_PATH_SQL)
        cursor = conn.cursor()

        # Execute the query
        cursor.execute(sql_query)

        # Fetch all the results
        rows = cursor.fetchall()

        # Get the column names
        columns = [description[0] for description in cursor.description]

        conn.commit()  # Commit any changes (if it's an INSERT/UPDATE/DELETE query)
        conn.close()

        # Return results as a list of dictionaries
        result = [dict(zip(columns, row)) for row in rows]
        return result
    except Exception as e:
        logging.error(f"Error executing query: {e}")
        return None
    

def get_table_columns(table_name: str):
    # Connect to the SQLite database
    conn = sqlite3.connect('your_database.db')  # Replace with your actual database path
    cursor = conn.cursor()

    # Use PRAGMA to get the table schema (column names)
    cursor.execute(f'PRAGMA table_info({table_name})')
    columns_info = cursor.fetchall()  # Fetch all rows returned by PRAGMA table_info

    # Extract column names from the fetched rows
    column_names = [column[1] for column in columns_info]  # column[1] contains the column name

    conn.close()

    return column_names

import pandas as pd

def get_csv_columns(csv_file_path: str):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_file_path)
    
    # Get the column names
    column_names = df.columns.tolist()

    return column_names

file_lock = threading.Lock()

@app.post("/query")
async def query_endpoint(request: Request):
    try:
        data = await request.json()
        natural_language_query = data.get('query')

        if not natural_language_query:
            raise HTTPException(status_code=400, detail="No query provided.")
        
        logging.info(f"Received query: {natural_language_query}")

        with file_lock:
            with open("user_queries.log", "a") as f:
                f.write(f"{datetime.now()} - User Query: {natural_language_query}\n")

        # Step 1: Get the embedding of the query using the /word2vec endpoint
        embedding_service_url = "http://localhost:8003/word2vec"  # Adjust the port if necessary
        try:
            embedding_response = requests.post(
                embedding_service_url,
                json={"text": natural_language_query}
            )
            embedding_response.raise_for_status()
            embedding = embedding_response.json().get("embedding")
        except requests.RequestException as e:
            logging.error(f"Error obtaining embedding: {e}")
            raise HTTPException(status_code=500, detail=f"Error obtaining embedding: {e}")

        if not embedding:
            raise HTTPException(status_code=500, detail="No embedding returned from /word2vec service.")
        
        logging.info(f"Embedding retrieved: {embedding}")

        # Step 2: Use the embedding to get the table_name and data_source from /queryembedding endpoint
        query_embedding_service_url = "http://localhost:8002/queryembedding"  # Adjust the port if necessary
        try:
            query_embedding_response = requests.post(
                query_embedding_service_url,
                json={"embedding": embedding}
            )
            query_embedding_response.raise_for_status()
            result = query_embedding_response.json()
            table_name = result.get("table_name")
            data_source = result.get("data_source")
        except requests.RequestException as e:
            logging.error(f"Error obtaining table name and data source: {e}")
            raise HTTPException(status_code=500, detail=f"Error obtaining table name and data source: {e}")

        if not table_name:
            raise HTTPException(status_code=500, detail="No table name returned from /queryembedding service.")

        logging.info(f"Table name: {table_name}, Data source: {data_source}")


        if data_source == 'SQL' :
            table_columns = get_table_columns(table_name)
        else :
            table_columns = get_csv_columns(csv_file_path)

        
       
        payload = {
            "question": natural_language_query,
            "tables": {
                table_name: table_columns  # Pass the table and columns in the format required
            }
        }
        payload = json.dumps(payload)

        print("payload is here -------------->",payload)
        # Send POST request to model service for inference
        MODEL_SERVICE_URL = "http://localhost:8000/inference"
        
        response = requests.post(MODEL_SERVICE_URL, data=payload, headers={"Content-Type": "application/json"})
        
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail="Error during inference")

        # Extract the SQL query from the model's response
        response_data = response.json()
        sql_query = response_data.get("sql_query")

        if not sql_query:
            raise HTTPException(status_code=500, detail="No SQL query returned from the model service.")

        logging.info(f"Generated SQL Query: {sql_query}")

        with file_lock:
            with open("user_queries.log", "a") as f:
                f.write(f"{datetime.now()} - Generated SQL Query: {sql_query}\n")

        # Return the SQL query\
        if data_source == 'SQL':
            query_result = execute_sql_query(sql_query)
        else :
            query_result = run_sql_on_csv(str(sql_query))

        if query_result is None:
            raise HTTPException(status_code=500, detail="Error executing the SQL query.")

        # Return the actual data from the query
        return {"result": query_result}

    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")
    


