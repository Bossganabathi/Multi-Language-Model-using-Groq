import os
import streamlit as st
from groq import Groq

st.set_page_config(
    page_title="Multi Model Application",
    page_icon="🤖",
    layout="centered"
)

from dotenv import load_dotenv

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

client = Groq()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("Multi Model Application")

with st.sidebar:
    st.title("Multi Model Application")
    models = ["mixtral-8x7b-32768", "llama-3.1-8b-instant", "Gemma-7b-it"]
    selected_model = st.selectbox("Select a model", models)

st.write(f"Selected model: {selected_model}")

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
user_prompt = st.chat_input("Enter your prompt:")

if user_prompt:
    
    st.chat_message("user").markdown(user_prompt)
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        *st.session_state.chat_history,
    ]
    
    response = client.chat.completions.create(
        model=selected_model,
        messages=messages,
        
    )
    
    assistant_response = response.choices[0].message.content
    st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
    
    with st.chat_message("assistant"):
        st.markdown(assistant_response)
    
    
        