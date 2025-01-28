# Agentic RAG System

An intelligent RAG (Retrieval Augmented Generation) system that uses an LLM agent to make decisions about information retrieval and response generation. The system processes PDF documents and can intelligently decide which knowledge base to query based on the user's question.

The system has the following features:

- Intelligent query routing
- PDF processing using Docling for accurate text extraction
- Persistent vector storage with ChromaDB
- Smart context retrieval and response generation
- FastAPI-based REST API
- Support for both OpenAI-based agents or local, transformer-based agents (Mistral-7B by default)

## Setup

1. Clone the repository and install dependencies:

    ```bash
    git clone https://github.com/oracle-devrel/devrel-labs.git
    cd agentic-rag
    pip install -r requirements.txt
    ```

2. Authenticate with HuggingFace:
   
   The system uses Mistral-7B by default, which requires authentication with HuggingFace:

   a. Create a HuggingFace account [here](https://huggingface.co/join)
   
   b. Accept the Mistral-7B model terms & conditions [here](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)
   
   c. Create an access token [here](https://huggingface.co/settings/tokens)
   
   d. Create a `config.yaml` file (you can copy from `config_example.yaml`):
   ```yaml
   HUGGING_FACE_HUB_TOKEN: your_token_here
   ```

3. (Optional) If you want to use the OpenAI-based agent instead of the default local model, create a `.env` file with your OpenAI API key:

   ```bash
   OPENAI_API_KEY=your-api-key-here
   ```

4. If no API key is provided, the system will automatically download and use `Mistral-7B-Instruct-v0.2` for text generation when using the local model. No additional configuration is needed.
   
## 1. Getting Started

You can use this solution in three ways:

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
python pdf_processor.py --input path/to/document.pdf --output chunks.json

# Process multiple PDFs in a directory
python pdf_processor.py --input path/to/pdf/directory --output chunks.json

# Process a single PDF from a URL 
python pdf_processor.py --input https://example.com/document.pdf --output chunks.json
```

#### Manage Vector Store

Add documents to the vector store and query them:
```bash
# Add documents from a chunks file
python store.py --add chunks.json --store-path my_chroma_db

# Query the vector store
python store.py --query "your search query" --store-path my_chroma_db
```

#### Use RAG Agent
Query documents using either the OpenAI or local model:
```bash
# Using OpenAI (requires API key in .env)
python rag_agent.py --query "What are the main topics?" --store-path my_chroma_db

# Using local Mistral model
python local_rag_agent.py --query "What are the main topics?" --store-path my_chroma_db
```

### 3. Complete Pipeline Example

Here's how to process a document and query it using the local model:
```bash
# 1. Process the PDF
python pdf_processor.py --input example.pdf --output chunks.json

# 2. Add to vector store
python store.py --add chunks.json --store-path my_chroma_db

# 3. Query using local model
python local_rag_agent.py --query "What is the main conclusion?" --store-path my_chroma_db
```

Or using OpenAI (requires API key):
```bash
# Same steps 1 and 2 as above, then:
python rag_agent.py --query "What is the main conclusion?" --store-path my_chroma_db
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

## Annex: Architecture

The system consists of several key components:

1. **PDF Processor**: we use Docling to extract and chunk text from PDF documents
2. **Vector Store**: Manages document embeddings and similarity search using ChromaDB
3. **RAG Agent**: Makes intelligent decisions about query routing and response generation
   - OpenAI Agent: Uses `gpt-4-turbo-preview` for high-quality responses, but requires an OpenAI API key
   - Local Agent: Uses `Mistral-7B` as an open-source alternative
4. **FastAPI Server**: Provides REST API endpoints for document upload and querying

## Hardware Requirements

- For OpenAI Agent: Standard CPU machine
- For Local Agent: 
  - Minimum 16GB RAM, recommended more than 24GBs
  - GPU with 8GB VRAM recommended for better performance
  - Will run on CPU if GPU is not available, but will be significantly slower.

TODO: integrate with Trafilatura to crawl web content apart from PDF

## Contributing

This project is open source. Please submit your contributions by forking this repository and submitting a pull request! Oracle appreciates any contributions that are made by the open source community.

## License

Copyright (c) 2024 Oracle and/or its affiliates.

Licensed under the Universal Permissive License (UPL), Version 1.0.

See [LICENSE](../LICENSE) for more details.

ORACLE AND ITS AFFILIATES DO NOT PROVIDE ANY WARRANTY WHATSOEVER, EXPRESS OR IMPLIED, FOR ANY SOFTWARE, MATERIAL OR CONTENT OF ANY KIND CONTAINED OR PRODUCED WITHIN THIS REPOSITORY, AND IN PARTICULAR SPECIFICALLY DISCLAIM ANY AND ALL IMPLIED WARRANTIES OF TITLE, NON-INFRINGEMENT, MERCHANTABILITY, AND FITNESS FOR A PARTICULAR PURPOSE. FURTHERMORE, ORACLE AND ITS AFFILIATES DO NOT REPRESENT THAT ANY CUSTOMARY SECURITY REVIEW HAS BEEN PERFORMED WITH RESPECT TO ANY SOFTWARE, MATERIAL OR CONTENT CONTAINED OR PRODUCED WITHIN THIS REPOSITORY. IN ADDITION, AND WITHOUT LIMITING THE FOREGOING, THIRD PARTIES MAY HAVE POSTED SOFTWARE, MATERIAL OR CONTENT TO THIS REPOSITORY WITHOUT ANY REVIEW. USE AT YOUR OWN RISK.