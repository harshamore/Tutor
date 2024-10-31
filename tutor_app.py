import streamlit as st
import replicate
from duckduckgo_search import DDGS
from datetime import datetime

# Set up Replicate API client
replicate_client = replicate.Client(api_token=st.secrets["replicate"]["api_key"])

ddgs = DDGS()

# Search the web for the given query
def search_web(query):
    print(f"Searching the web for {query}...")
    
    # DuckDuckGo search
    current_date = datetime.now().strftime("%Y-%m")
    results = ddgs.text(f"{query} {current_date}", max_results=10)
    if results:
        news_results = ""
        for result in results:
            news_results += f"Title: {result['title']}\nURL: {result['href']}\nDescription: {result['body']}\n\n"
        return news_results.strip()
    else:
        return f"Could not find news results for {query}."

# Run the workflow using Replicate's Llama model
def run_workflow(query):
    print("Running web research assistant workflow...")
    
    # Step 1: Search the web
    raw_news = search_web(query)

    # Step 2: Summarize search results using Replicate Llama model
    try:
        # Call the Llama-2-7b-chat model on Replicate for summarization
        research_analysis = replicate_client.run(
            "meta/llama-2-7b-chat",  # Replace with actual model version if needed
            input={"text": raw_news}
        )
        
        # Use Llama-2-7b-chat model again to generate a polished, publication-ready article
        final_article = replicate_client.run(
            "meta/llama-2-7b-chat",  # Same model used again, or replace with any different model for editing
            input={"text": research_analysis}
        )

        return final_article

    except Exception as e:
        return f"Error processing workflow with Replicate Llama model: {e}"

# Streamlit app
def main():
    st.set_page_config(page_title="Internet Research Assistant 🔎", page_icon="🔎")
    st.title("Internet Research Assistant 🔎")

    # Initialize session state for query and article
    if 'query' not in st.session_state:
        st.session_state.query = ""
    if 'article' not in st.session_state:
        st.session_state.article = ""

    # Create two columns for the input and clear button
    col1, col2 = st.columns([3, 1])

    # Search query input
    with col1:
        query = st.text_input("Enter your search query:", value=st.session_state.query)

    # Clear button
    with col2:
        if st.button("Clear"):
            st.session_state.query = ""
            st.session_state.article = ""
            st.rerun()

    # Generate article only when button is clicked
    if st.button("Generate Article") and query:
        with st.spinner("Generating article..."):
            # Get response from Replicate Llama model
            article = run_workflow(query)
            st.session_state.query = query
            st.session_state.article = article

    # Display the article if it exists in the session state
    if st.session_state.article:
        st.markdown(st.session_state.article)

if __name__ == "__main__":
    main()
