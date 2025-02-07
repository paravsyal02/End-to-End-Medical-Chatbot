# End-to-End-Medical-Chatbot

# Medical Chatbot

This is a simple medical chatbot application built using Flask, LangChain, Google Gemini AI, and Pinecone for document retrieval. It allows users to ask medical-related questions, and the chatbot provides relevant answers based on pre-loaded medical data using embeddings stored in a Pinecone vector database.

## Features

- User can ask medical-related questions.
- The chatbot retrieves information from a Pinecone vector store containing embeddings of medical documents.
- Uses Google Gemini AI for generating responses based on the retrieved context.
- Responsive chat interface with a simple and clean design.

## Requirements

- Python 3.8+
- Install dependencies using `pip`:

```bash
pip install -r requirements.txt
