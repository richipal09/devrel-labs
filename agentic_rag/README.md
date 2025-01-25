# Agentic RAG System

An intelligent RAG (Retrieval Augmented Generation) system that uses an LLM agent to make decisions about information retrieval and response generation. The system processes PDF documents and can intelligently decide which knowledge base to query based on the user's question.

## Features

- 🤖 Intelligent query routing using an LLM agent
- 📄 PDF processing using Docling for accurate text extraction
- 💾 Persistent vector storage using ChromaDB
- 🔍 Smart context retrieval and response generation
- 🚀 FastAPI-based REST API
- 🌟 Support for both OpenAI and local open-source models

## Setup

1. Clone the repository and install dependencies:

```bash
git clone <repository-url>
cd agentic-rag
pip install -r requirements.txt
```

2. Choose your LLM configuration:

   **Option 1: OpenAI (requires API key)**
   Create a `.env` file with your OpenAI API key:
   ```bash
   OPENAI_API_KEY=your-api-key-here
   ```

   **Option 2: Local Open-Source Model**
   No additional configuration needed. The system will automatically use Mistral-7B-Instruct-v0.2.

## Usage Options

You can use the system in three ways:

### 1. Using the Complete REST API

Start the API server:
```bash
python main.py
```

The API will be available at `http://localhost:8000`. You can then use the API endpoints as described in the API Endpoints section below.

### 2. Using Individual Components via Command Line

#### Process PDFs
Process a PDF file and save the chunks to a JSON file:
```bash
# Process a single PDF
python -m document_processor.pdf_processor --input path/to/document.pdf --output chunks.json

# Process multiple PDFs in a directory
python -m document_processor.pdf_processor --input path/to/pdf/directory --output chunks.json
```

#### Manage Vector Store
Add documents to the vector store and query them:
```bash
# Add documents from a chunks file
python -m vector_store.store --add chunks.json --store-path my_chroma_db

# Query the vector store
python -m vector_store.store --query "your search query" --store-path my_chroma_db
```

#### Use RAG Agent
Query documents using either the OpenAI or local model:
```bash
# Using OpenAI (requires API key in .env)
python -m agents.rag_agent --query "What are the main topics?" --store-path my_chroma_db

# Using local Mistral model
python -m agents.local_rag_agent --query "What are the main topics?" --store-path my_chroma_db
```

### 3. Complete Pipeline Example

Here's how to process a document and query it using the local model:
```bash
# 1. Process the PDF
python -m document_processor.pdf_processor --input example.pdf --output chunks.json

# 2. Add to vector store
python -m vector_store.store --add chunks.json --store-path my_chroma_db

# 3. Query using local model
python -m agents.local_rag_agent --query "What is the main conclusion?" --store-path my_chroma_db
```

Or using OpenAI (requires API key):
```bash
# Same steps 1 and 2 as above, then:
python -m agents.rag_agent --query "What is the main conclusion?" --store-path my_chroma_db
```

## API Endpoints

### Upload PDF

```http
POST /upload/pdf
Content-Type: multipart/form-data

file: <pdf-file>
```

Uploads and processes a PDF file, storing its contents in the vector database.

### Query

```http
POST /query
Content-Type: application/json

{
    "query": "your question here"
}
```

Processes a query through the agentic RAG pipeline and returns a response with context.

## Architecture

The system consists of several key components:

1. **PDF Processor**: Uses Docling to extract and chunk text from PDF documents
2. **Vector Store**: Manages document embeddings and similarity search using ChromaDB
3. **RAG Agent**: Makes intelligent decisions about query routing and response generation
   - OpenAI Agent: Uses GPT-4 for high-quality responses (requires API key)
   - Local Agent: Uses Mistral-7B for open-source alternative
4. **FastAPI Server**: Provides REST API endpoints for document upload and querying

## Hardware Requirements

- For OpenAI Agent: Standard CPU machine
- For Local Agent: 
  - Minimum 16GB RAM
  - GPU with 8GB VRAM recommended for better performance
  - Will run on CPU if GPU is not available (slower)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 