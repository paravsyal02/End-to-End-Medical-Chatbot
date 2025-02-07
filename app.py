from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

embeddings = download_hugging_face_embeddings()


index_name= "medicalbot"

#Embed each chunk and upsert the embeddings into your Pinecone Index
docsearch = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings
)

retriever = docsearch.as_retriever()


llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.6,
    timeout=None,
    max_retries=2,
    api_key=GOOGLE_API_KEY, 
)

prompt = ChatPromptTemplate.from_messages(
    [
      ("system", system_prompt),
      ("human", "{input}"),

    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])  
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    retrieved_docs = retriever.invoke(msg)
    response = rag_chain.invoke({"input": msg, "context": retrieved_docs})
    print("Response : ", response["answer"])
    return str(response["answer"])



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)