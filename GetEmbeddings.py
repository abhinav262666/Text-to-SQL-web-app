from fastapi import FastAPI
from pydantic import BaseModel
from gensim.models import KeyedVectors
import numpy as np

# Initialize FastAPI app
app = FastAPI()

# Load the pre-trained Word2Vec model (Google's pre-trained Word2Vec model)
# Make sure to replace this path with the actual path to your downloaded model
model_path = r"E:\ProcureyardAssignment\ai-engineer-explorer-abhinav262666\GoogleNews-vectors-negative300.bin\GoogleNews-vectors-negative300.bin"
model = KeyedVectors.load_word2vec_format(model_path, binary=True)

# Define the input model for the request
class TextData(BaseModel):
    text: str

# Function to get average Word2Vec embeddings for the input text
def get_word2vec_embedding(text):
    words = text.split()
    valid_words = [word for word in words if word in model]

    # If none of the words are in the model, return a zero vector
    if not valid_words:
        return np.zeros(model.vector_size)

    # Get the embeddings for each valid word and average them
    word_vectors = [model[word] for word in valid_words]
    avg_embedding = np.mean(word_vectors, axis=0)
    
    return avg_embedding

# Define the POST endpoint to get Word2Vec embeddings
@app.post("/word2vec")
async def get_word2vec(text_data: TextData):
    # Get the input text from the request
    text = text_data.text

    # Generate the average Word2Vec embeddings
    embeddings = get_word2vec_embedding(text)

    # Convert the embeddings to a list for JSON serialization
    embeddings_list = embeddings.tolist()

    # Return the embeddings as a JSON response
    return {"embedding": embeddings_list}
