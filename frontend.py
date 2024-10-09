import streamlit as st
import requests

# Define FastAPI endpoint
API_ENDPOINT = "http://127.0.0.1:8001/query"

# Set up the Streamlit app
st.title("Natural Language to SQL Query Generator")

# Collect the user's natural language query input
user_query = st.text_input("Enter your query in natural language:")

# Button to submit the query
if st.button("Generate SQL Query"):
    if user_query:
        # Send the query to the FastAPI endpoint
        response = requests.post(API_ENDPOINT, json={"query": user_query})

        if response.status_code == 200:
            # Display the result
            result = response.json().get("result")
            st.write("Query Result:")
            
            if result:
                # Display the result in a clean format
                for item in result:
                    for key, value in item.items():
                        st.write(f"{key}: {value}")
            else:
                st.write("No results found.")
        else:
            st.error("Error in generating SQL query.")
    else:
        st.warning("Please enter a query.")
