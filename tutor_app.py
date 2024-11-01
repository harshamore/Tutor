import os
import streamlit as st
import replicate
from requests.exceptions import ReadTimeout

# Set up REPLICATE_API_TOKEN environment variable from Streamlit secrets
os.environ["REPLICATE_API_TOKEN"] = st.secrets["REPLICATE_API_TOKEN"]

# Initialize Replicate client
replicate_client = replicate.Client()

# Initialize chat history in session state if it doesn't already exist
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Function to get response from Llama model via Replicate with timeout handling
def get_llama_response(user_input, max_retries=3):
    prompt = f"HARVARD MBA Professor answering business questions in a simplified way with real world example in less than 200 words:\n{user_input}"
    input_params = {
        "top_k": 250,
        "prompt": prompt,
        "temperature": 0.95,
        "max_new_tokens": 200,  # Adjust token count as needed
    }
    
    response = ""
    for attempt in range(max_retries):
        try:
            # Call Replicate's model and capture response as a single batch
            response = replicate_client.run("meta/meta-llama-3-70b-instruct", input=input_params)
            break  # Exit loop if successful
        except ReadTimeout:
            if attempt < max_retries - 1:
                st.warning(f"Timeout occurred, retrying... ({attempt + 1}/{max_retries})")
            else:
                response = "Error: The operation timed out after multiple attempts."
        except Exception as e:
            response = f"Error: {e}"
            break
    
    return response

# Streamlit app interface
st.title("Professor Mrs. Singy Fonty")

# Input box for user message, directly stored in session state
if "user_input" not in st.session_state:
    st.session_state.user_input = ""  # Initialize if not already set

# Text input updates session state directly
st.session_state.user_input = st.text_input("Your message:", value=st.session_state.user_input)

# Display chat history
for i, (user, bot) in enumerate(st.session_state.chat_history):
    st.write(f"**User:** {user}")
    st.write(f"**Professor Mrs. Singy Fonty:** {bot}")

# Process user input and generate response when "Send" button is clicked
if st.button("Send") and st.session_state.user_input.strip():
    # Store user input and clear the input field in session state
    user_input = st.session_state.user_input
    st.session_state.user_input = ""  # Clear input field for next message

    # Append user input to chat history with a placeholder for the response
    st.session_state.chat_history.append((user_input, "Generating response..."))

    # Get response from the model
    llama_response = get_llama_response(user_input)
    
    # Update the latest chat entry with the model's response
    st.session_state.chat_history[-1] = (user_input, llama_response)
