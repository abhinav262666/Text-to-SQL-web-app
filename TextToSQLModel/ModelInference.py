from fastapi import FastAPI, Request
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

app = FastAPI()

# Load the model and tokenizer once when the app starts
tokenizer = AutoTokenizer.from_pretrained("juierror/flan-t5-text2sql-with-schema-v2")
model = AutoModelForSeq2SeqLM.from_pretrained("juierror/flan-t5-text2sql-with-schema-v2")

# Move the model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Helper function to create a formatted prompt
def get_prompt(tables, question):
    prompt = f"convert question and table into SQL query. tables: {tables}. question: {question}"
    return prompt

def prepare_input(question: str, tables: dict):
    # Convert the tables dict to the required string format like table_name(column1, column2, ...)
    tables_str = [f"{table_name}({', '.join(columns)})" for table_name, columns in tables.items()]
    tables_str = ", ".join(tables_str)  # Join all tables as a single string
    
    # Create the prompt for the model
    prompt = get_prompt(tables_str, question)
    
    # Tokenize the prompt
    input_ids = tokenizer(prompt, max_length=512, return_tensors="pt").input_ids.to(device)
    return input_ids

@app.post("/inference")
async def inference(request: Request):
    data = await request.json()
    question = data.get("question")
    tables = data.get("tables")

    # Error handling: Make sure the question and tables are provided
    if not question or not tables:
        return {"error": "Missing question or tables"}

    # Prepare input for the model
    input_ids = prepare_input(question, tables)

    # Generate SQL query
    outputs = model.generate(inputs=input_ids, num_beams=10, top_k=10, max_length=512)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Return the generated SQL query
    return {"sql_query": result}
