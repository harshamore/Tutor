import streamlit as st
import replicate

# Set up Replicate API client with API key from secrets
replicate_client = replicate.Client(api_token=st.secrets["replicate"]["api_key"])

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Function to get response from Llama model via Replicate
def get_llama_response(user_input):
    # Prefix user input with the specified role
    prompt = f"HARVARD MBA Professor:\n{user_input}"
    input_params = {
        "top_k": 250,
        "prompt": prompt,
        "temperature": 0.95,
        "max_new_tokens": 500,
        "min_new_tokens": -1
    }
    
    # Stream response from Replicate API
    response = ""
    try:
        for event in replicate.stream("meta/llama-2-7b", input=input_params):
            response += event
            # Update the response in real-time
            chat_placeholder.markdown(response + "▌")
    except Exception as e:
        response = f"Error: {e}"
    
    return response

# Streamlit app interface
st.title("Chat with Llama-2-7b - HARVARD MBA Professor Mode")

# Chat input box
user_input = st.text_input("Your message:", key="user_input")

# Display chat history
for i, (user, bot) in enumerate(st.session_state.chat_history):
    st.write(f"**User:** {user}")
    st.write(f"**Llama-2-7b (HARVARD MBA Professor):** {bot}")

# Process user input and generate response
if st.button("Send") and user_input:
    # Append user input to chat history
    st.session_state.chat_history.append((user_input, ""))  # Temporary empty response
    user_message_index = len(st.session_state.chat_history) - 1
    
    # Display real-time response
    chat_placeholder = st.empty()
    
    # Get response from model
    llama_response = get_llama_response(user_input)
    
    # Update chat history with model's response
    st.session_state.chat_history[user_message_index] = (user_input, llama_response)
    
    # Clear the input field
    st.session_state.user_input = ""
