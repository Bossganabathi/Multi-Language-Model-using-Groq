import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS #vectorstore db
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings #vector embedding techniques


from dotenv import load_dotenv

load_dotenv()

## load the groq and google api key from the .env file

groq_api_key = os.getenv("GROQ_API_KEY")
os.environ['GOOGLE_API_KEY'] = os.getenv("GOOGLE_API_KEY")

st.title("Multi Model Question Generator")

with st.sidebar:
    st.title("Multi Model Application")
    models = ["mixtral-8x7b-32768", "llama-3.1-8b-instant", "Gemma-7b-it", "Gemma2-9b-it"]
    selected_model = st.selectbox("Select a model", models)

st.write(f"Selected model: {selected_model}")

llm = ChatGroq(groq_api_key=groq_api_key, model_name=selected_model, temperature=0.5)

prompt = ChatPromptTemplate.from_template(
    """
    Generate questions answers based on text. The questions generated able to answer and get from text, the questions should have answers. Each questions must have 4 choice of answer. Questions and answers must in BAHASA MELAYU language.
    <context>
    {context}
    <context>
    Questions:{input}
    
    """
)
from langchain_community.document_loaders import PyPDFLoader

def vector_embedding(file_path):
    if "vectors" not in st.session_state:
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        st.session_state.loader=PyPDFLoader(file_path) #data ingestion
        st.session_state.docs = st.session_state.loader.load() #document loading
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:20])
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)

prompt1 = st.text_input("What you want to ask from the documents?")

uploaded_file = st.file_uploader("Upload a file in PDF Format", type=["pdf"])

if uploaded_file is not None:
    with open("uploaded_file.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success("File uploaded successfully!")


if st.button("Creating Vector Store"):
    vector_embedding("uploaded_file.pdf")
    st.write("Vector Store DB IS Ready")
    
import time

if prompt1:
    document_chain=create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    start = time.process_time()
    response = retrieval_chain.invoke({'input':prompt1})
    print("Response time :",time.process_time()-start)
    st.write(response['answer'])
    #print(response)
    
    with st.expander("Document Similarity Search"):
        # Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")
            
    