import os
import streamlit as st

# Ensure that HF_TOKEN is loaded from the secrets file into the environment, whether in .env or streamlit config.
if "HF_TOKEN" not in os.environ:
    try:
        os.environ["HF_TOKEN"] = st.secrets["HF_TOKEN"]
    except Exception as e:
        st.error("HF_TOKEN not found in secrets. Please set HF_TOKEN in .streamlit/secrets.toml.")
        raise e

from multimodal.src.api_call import ApiCalls

# Initialize the API calls module (this will also load the environment variables via LLMClient)
api = ApiCalls()

st.title("Multimodal LLM Application")

# Sidebar to choose the function
function = st.sidebar.selectbox(
    "Select a function", 
    ["Entity Recognition", "Image to Text", "Advanced Image to Text", "Summarize & Key Points", "RAG Query"]
)

# Input area for parameters
if function == "Entity Recognition":
    text_input = st.text_area("Enter text for entity recognition", "My name is Sarah Jessica Parker but you can call me Jessica")
elif function == "Image to Text":
    option = st.radio("Image source:", ("Local file", "URL"))
    if option == "Local file":
        image_path = st.text_input("Enter local file path", "multimodal/data/demo.jpg")
    else:
        image_path = st.text_input("Enter image URL", "https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg")
elif function == "Advanced Image to Text":
    option = st.radio("Image source:", ("Local file", "URL"))
    if option == "Local file":
        image_path = st.text_input("Enter local file path", "multimodal/data/demo.jpg")
    else:
        image_path = st.text_input("Enter image URL", "https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg")
elif function == "Summarize & Key Points":
    text_input = st.text_area("Enter text to summarize", "Paste long text here...")
elif function == "RAG Query":
    text_input = st.text_input("Enter your question about LLMs and VLMs", "What is long-horizon planning, and reward design?")

# Run button
if st.button("Run"):
    try:
        if function == "Entity Recognition":
            result = api.entity_recognition(text_input)
        elif function == "Image to Text":
            result = api.img2text(image_path)
        elif function == "Advanced Image to Text":
            result = api.advanced_img2text(image_path)
        elif function == "Summarize & Key Points":
            result = api.summarize_and_key_points(text_input)
        elif function == "RAG Query":
            result = api.RAG_query(text_input)
        st.success("Result:")
        st.write(result)
    except Exception as e:
        st.error(f"Error: {str(e)}")