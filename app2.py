import streamlit as st
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize embeddings
embeddings = download_hugging_face_embeddings()

# Define Pinecone index name
index_name = "medicalbot"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)
retriever = docsearch.as_retriever()

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.6,
    timeout=None,
    max_retries=2,
    api_key=GOOGLE_API_KEY  # Ensure you're using environment variables, not hardcoding API keys
)

# Define the prompt
system_prompt = """
    You are an assistant for question-answering tasks.
    Use the following retrieved context to answer the question.
    If you don't know the answer, say that you don't know.
    Keep the answer detailed yet concise.
    \n\n
    {context}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# Create the retrieval-augmented generation (RAG) chain
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Streamlit UI
st.set_page_config(page_title="Medical Chatbot", layout="wide")
st.title("🩺 AI-Powered Medical Chatbot")
st.write("Ask me any medical-related questions!")

# Chat input
user_input = st.text_input("Type your question here:", "")

if st.button("Ask AI") and user_input:
    retrieved_docs = retriever.invoke(user_input)
    response = rag_chain.invoke({"input": user_input, "context": retrieved_docs})
    
    st.subheader("Chatbot Response:")
    st.write(response["answer"])
