# Natural Language to SQL Query Generator

This project allows users to convert natural language queries into SQL queries and execute them on either CSV or SQL databases. The application is built using FastAPI for the backend and Streamlit for the frontend, with support for querying both SQL databases and CSV files.

## Table of Contents

- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [How to Run](#how-to-run)


## Project Structure

- **main.py**: The main logic to process the user query, generate SQL, and execute it.
- **frontend.py**: The Streamlit frontend to interact with users.
- **GetEmbeddings.py**: Loads a word2vec model to create and send embeddings.
- **StoreAndCreateEmbeddings.py**: Logic to determine the data source and table name using embeddings.
- **ModelInference.py**: A FastAPI service for generating SQL queries from natural language using a model from Hugging Face.
- **your_database.db**: SQLite database containing SQL tables.
- **Sales_Data.csv**: CSV file used as a data source.
- **Dockerfile**: Builds a Docker image for the Hugging Face model used in SQL generation.
- **docker-compose.yml**: Docker Compose configuration to manage different services.
- **requirements.txt**: Python dependencies required for the project.

## Prerequisites

- Python 3.9 or higher
- Docker (for building and running images)
- Git (for cloning the repository)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   
2. Create and activate a virtual environment (optional but recommended):
    ```bash
   python -m venv myenv
   source myenv/bin/activate   # On Windows use `myenv\Scripts\activate`

4. Install the required dependencies:
   ```bash
   pip install -r requirements.txt

6. Set up the necessary files:
  Ensure your_database.db is present for SQLite database queries.
  Ensure Sales_Data.csv is present for CSV data queries.

7. Download the pre-trained Word2Vec model from the following link: [Download Word2Vec Model](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?resourcekey=0-wjGZdNAUop6WykTtMip30g).
   Place the downloaded model in the appropriate folder as referenced in GetEmbeddings.py.


## How to run 

1. Start the FastAPI backend (for query processing):
   ```bash
   uvicorn main:app --reload --port 8001

2. Run the Streamlit frontend::
   ```bash
   streamlit run frontend.py

3. Run Embeddings Service (GetEmbeddings.py):
   ```bash
   uvicorn GetEmbeddings:app --reload --port 8003

4. Run Embeddings Retrieval Service (StoreAndCreateEmbeddings.py):
   ```bash
   uvicorn StoreAndCreateEmbeddings.app --reload --port 8002

5. Docker for the Hugging Face Model Inference
  To build the Docker image for the model that converts natural language to SQL, use the following commands:
  ```bash
   docker build -t texttosqlinference .
   docker run -p 8000:8000 texttosqlinference
# Text-to-SQL-web-app
# Text-to-SQL-web-app
