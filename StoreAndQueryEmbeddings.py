# import faiss
# import numpy as np
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# import uvicorn

# # Initialize FastAPI app
# app = FastAPI()

# # Metadata store: for each vector in FAISS, we'll associate it with table_name and data_source (SQL/CSV)
# metadata = []

# # Step 1: Initialize FAISS index
# # Assuming you're using 300-dimensional embeddings (like Word2Vec embeddings)
# dimension = 300
# index = faiss.IndexFlatL2(dimension)  # L2 distance index (Euclidean)

# # Define Pydantic model for incoming requests
# class EmbeddingData(BaseModel):
#     embedding: list  # The embedding vector, a list of floats
#     table_name: str  # The table name associated with the embedding
#     data_source: str  # Whether it's from SQL or CSV

# class QueryData(BaseModel):
#     embedding: list  # The embedding to search in the FAISS index

# # Function to add an embedding and associated metadata to the FAISS index
# def add_to_faiss(vector, table_name, data_source):
#     # Convert the vector to a numpy array and reshape it to the correct shape
#     vector_np = np.array(vector).astype('float32').reshape(1, -1)
    
#     # Add the vector to the FAISS index
#     index.add(vector_np)
    
#     # Store the metadata associated with this vector
#     metadata.append({"table_name": table_name, "data_source": data_source})

# # Step 2: Add an API endpoint to add new embeddings via POST
# @app.post("/add_embedding")
# async def add_embedding(embedding_data: EmbeddingData):
#     # Validate that the embedding has the correct dimension
#     if len(embedding_data.embedding) != dimension:
#         raise HTTPException(status_code=400, detail=f"Embedding must be {dimension} dimensions.")
    
#     # Add the embedding and associated metadata to the FAISS index
#     add_to_faiss(embedding_data.embedding, embedding_data.table_name, embedding_data.data_source)
    
#     return {"message": "Embedding added successfully", "table_name": embedding_data.table_name, "data_source": embedding_data.data_source}

# # Step 3: FastAPI endpoint to query the FAISS index
# @app.post("/queryembedding")
# async def query_embedding(query_data: QueryData):
#     # Convert the incoming embedding to a numpy array
#     query_vector = np.array(query_data.embedding).astype('float32').reshape(1, -1)

#     # Check if the FAISS index contains vectors
#     if index.ntotal == 0:
#         raise HTTPException(status_code=404, detail="No embeddings in the FAISS index.")

#     # Query FAISS for the nearest neighbor (k=1)
#     distances, indices = index.search(query_vector, 1)  # k=1 means we want the closest match

#     # Retrieve the index of the closest vector
#     closest_index = indices[0][0]

#     # If the closest vector has a distance of infinity, it means no match was found
#     if closest_index == -1:
#         raise HTTPException(status_code=404, detail="No matching table found.")

#     # Retrieve the metadata associated with the closest vector
#     result_metadata = metadata[closest_index]

#     # Return the metadata (table name and source type)
#     return {"table_name": result_metadata["table_name"], "data_source": result_metadata["data_source"]}


import json
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity  # To compute cosine similarity
import os

# Initialize FastAPI app
app = FastAPI()

# Path to the JSON file that stores the embeddings
EMBEDDING_FILE = "embeddings.json"

# Load embeddings from the JSON file if it exists, otherwise initialize an empty dictionary
if os.path.exists(EMBEDDING_FILE):
    with open(EMBEDDING_FILE, "r") as f:
        embeddings_data = json.load(f)
else:
    embeddings_data = {}

# Define Pydantic models for incoming requests
class EmbeddingData(BaseModel):
    embedding: list  # The embedding vector, a list of floats
    table_name: str  # The table name associated with the embedding
    data_source: str  # Whether it's from SQL or CSV

class QueryData(BaseModel):
    embedding: list  # The embedding to search in the stored embeddings

# Save the embeddings back to the JSON file
def save_embeddings_to_file():
    with open(EMBEDDING_FILE, "w") as f:
        json.dump(embeddings_data, f)

# Function to add an embedding and associated metadata to the JSON file
def add_embedding_to_json(embedding, table_name, data_source):
    # Store the embedding and its metadata as a dictionary
    embeddings_data[table_name] = {
        "embedding": embedding,
        "data_source": data_source
    }
    # Save to file
    save_embeddings_to_file()

# Step 2: API endpoint to add new embeddings via POST
@app.post("/add_embedding")
async def add_embedding(embedding_data: EmbeddingData):
    # Check if the embedding already exists for this table_name
    if embedding_data.table_name in embeddings_data:
        raise HTTPException(status_code=400, detail="Embedding for this table already exists.")

    # Add the embedding to the JSON file
    add_embedding_to_json(embedding_data.embedding, embedding_data.table_name, embedding_data.data_source)

    return {"message": "Embedding added successfully", "table_name": embedding_data.table_name, "data_source": embedding_data.data_source}

# Step 3: API endpoint to query the embeddings using cosine similarity
@app.post("/queryembedding")
async def query_embedding(query_data: QueryData):
    if not embeddings_data:
        raise HTTPException(status_code=404, detail="No embeddings stored.")

    query_vector = np.array(query_data.embedding).astype('float32').reshape(1, -1)
    closest_table = None
    max_similarity = -1

    # Iterate over the stored embeddings to find the most similar one using cosine similarity
    for table_name, data in embeddings_data.items():
        stored_embedding = np.array(data["embedding"]).reshape(1, -1)
        similarity = cosine_similarity(query_vector, stored_embedding)[0][0]
        
        if similarity > max_similarity:
            max_similarity = similarity
            closest_table = table_name

    if closest_table is None:
        raise HTTPException(status_code=404, detail="No matching table found.")

    result_metadata = embeddings_data[closest_table]

    # Return the metadata (table name and source type)
    return {"table_name": closest_table, "data_source": result_metadata["data_source"], "similarity": max_similarity}
