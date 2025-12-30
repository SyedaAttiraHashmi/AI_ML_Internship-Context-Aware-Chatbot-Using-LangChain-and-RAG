Context-Aware Chatbot Using LangChain

Project Objectives

This project is a context-aware chatbot that can have conversations with users and remember the previous messages. It can also retrieve information from documents (like PDFs or any knowledge base) to answer questions accurately.

The chatbot is built using LangChain and FAISS for vector search and deployed using Streamlit for a simple web interface.

Features

Context Memory: Remembers conversation history so it can answer follow-up questions.

Document Retrieval: Searches PDFs in the data/ folder to find relevant information.

RAG (Retrieval-Augmented Generation): Combines retrieval and LLM generation for better answers.

Streamlit UI: Clean and interactive web interface for chatting.

Optional Context Display: Users can see the source content from documents.

How It Works

Load Documents: The chatbot loads all PDFs from the data/ folder.

Split Documents: Documents are split into smaller chunks for better searching.

Vector Store: Uses FAISS to store chunks as vectors for fast retrieval.

LLM: Uses ChatGroq as the language model to generate answers.

Chain: Combines the retriever and LLM into a chain that remembers chat history.

Streamlit Interface: Users type a question and get answers along with optional context.

Project Structure
Task4_Chatbot/
│── app.py           # Streamlit app for the chat UI
│── main.py          # Code to initialize LLM, chain, and vector store
│── data/            # Folder containing PDF documents
│── .env             # API key for Groq model
│── README.md

How to Run

Install dependencies:

uv pip install -r requirements.txt


Make sure your PDFs are in the data/ folder.

Set your Groq API key in .env:

GROQ_API_KEY=your_api_key_here


Run the Streamlit app:

uv run streamlit run app.py


Open the app in your browser:
http://localhost:8501

Usage

Type a question in the chat box.

The chatbot will answer using both conversation context and document content.

You can check “Show PDF context” in the sidebar to see which part of the document was used.

Tools & Libraries Used

Python 3.10+

LangChain – for building conversational AI with memory

FAISS – vector store for document retrieval

ChatGroq – Large language model for generating answers

Streamlit – web interface for chatbot

Conclusion

This project demonstrates how to build a context-aware chatbot using LangChain and RAG. The chatbot is capable of remembering conversations and answering questions using external documents.

It can be extended in the future to:

Add more document formats (Word, HTML, etc.)

Improve UI/UX with modern chat design

Integrate with other LLMs for better performance