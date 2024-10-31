import streamlit as st
import replicate

import replicate

input = {
    "top_k": 250,
    "prompt": "Harvard MBA professor",
    "temperature": 0.95,
    "max_new_tokens": 500,
    "min_new_tokens": -1
}

for event in replicate.stream(
    "meta/llama-2-7b",
    input=input
):
    print(event, end="")
#=> ", enter your question...
