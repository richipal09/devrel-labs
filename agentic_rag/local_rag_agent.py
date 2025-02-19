from typing import List, Dict, Any, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from pydantic import BaseModel, Field
from store import VectorStore
import argparse
import yaml
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

class QueryAnalysis(BaseModel):
    """Pydantic model for query analysis output"""
    query_type: str = Field(
        description="Type of query: 'pdf_documents', 'general_knowledge', or 'unsupported'"
    )
    reasoning: str = Field(
        description="Reasoning behind the query type selection"
    )
    requires_context: bool = Field(
        description="Whether the query requires additional context to answer"
    )

class LocalRAGAgent:
    def __init__(self, vector_store: VectorStore, model_name: str = "mistralai/Mistral-7B-Instruct-v0.2", use_cot: bool = False, language: str = "en"):
        """Initialize local RAG agent with vector store and local LLM"""
        self.vector_store = vector_store
        self.use_cot = use_cot
        self.language = language
        
        # Load HuggingFace token from config
        try:
            with open('config.yaml', 'r') as f:
                config = yaml.safe_load(f)
            token = config.get('HUGGING_FACE_HUB_TOKEN')
            if not token:
                raise ValueError("HUGGING_FACE_HUB_TOKEN not found in config.yaml")
        except Exception as e:
            raise Exception(f"Failed to load HuggingFace token from config.yaml: {str(e)}")
        
        # Load model and tokenizer
        print("\nLoading model and tokenizer...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            token=token
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
        
        # Create text generation pipeline
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.1,
            top_p=0.95,
            device_map="auto"
        )
        print("✓ Model loaded successfully")
    
    def _generate_text(self, prompt: str, max_length: int = 512) -> str:
        """Generate text using the local model"""
        result = self.pipeline(
            prompt,
            max_new_tokens=max_length,
            do_sample=True,
            temperature=0.1,
            top_p=0.95,
            return_full_text=False
        )[0]["generated_text"]
        
        return result.strip()
    
    def _analyze_query(self, query: str) -> QueryAnalysis:
        """Analyze the query to determine the best source of information"""
        prompt = f"""You are an intelligent agent that analyzes user queries to determine the best source of information.

    Analyze the following query and determine:
    1. Whether it should query the PDF documents collection or general knowledge collection
    2. Your reasoning for this decision
    3. Whether the query requires additional context to provide a good answer

    Query: {query}

    Provide your response in the following JSON format:
    {{
        "query_type": "pdf_documents OR general_knowledge OR unsupported",
        "reasoning": "your reasoning here",
        "requires_context": true OR false
    }}

    Response:"""
        
        try:
            response = self._generate_text(prompt)
            # Extract JSON from the response using string manipulation
            start_idx = response.find("{")
            end_idx = response.rfind("}") + 1
            if start_idx != -1 and end_idx != -1:
                json_str = response[start_idx:end_idx]
                return QueryAnalysis.model_validate_json(json_str)
            raise ValueError("Could not parse JSON from response")
        except Exception as e:
            # Default to PDF documents if parsing fails
            return QueryAnalysis(
                query_type="pdf_documents",
                reasoning="Defaulting to PDF documents due to parsing error",
                requires_context=True
            )
    
    def _generate_direct_response(self, query: str) -> Dict[str, Any]:
        """Generate a response directly from the LLM without context"""
        logger.info("Generating direct response from LLM without context...")
        
        if self.use_cot:
            prompt = f"""You are a helpful AI assistant. Please answer the following query using chain of thought reasoning.
First break down the problem into steps, then solve each step to arrive at the final answer.

Query: {query}

Let's think about this step by step:"""
        else:
            prompt = f"""You are a helpful AI assistant. Please answer the following query to the best of your ability.
If you're not confident about the answer, please say so.

Query: {query}

Answer:"""
        
        logger.info("Generating response using local model...")
        response = self._generate_text(prompt, max_length=1024)
        logger.info("Response generation complete")
        
        return {
            "answer": response,
            "context": []
        }

    def process_query(self, query: str) -> Dict[str, Any]:
        """Process a user query using the agentic RAG pipeline"""
        logger.info(f"Starting to process query: {query}")
        
        # Analyze the query
        logger.info("Analyzing query type and context requirements...")
        analysis = self._analyze_query(query)
        logger.info(f"Query analysis results:")
        logger.info(f"- Type: {analysis.query_type}")
        logger.info(f"- Requires context: {analysis.requires_context}")
        logger.info(f"- Reasoning: {analysis.reasoning}")
        
        # If query type is unsupported, use general knowledge
        if analysis.query_type == "unsupported":
            logger.info("Query type is unsupported, using general knowledge...")
            return self._generate_general_response(query)
        
        # First try to get context from PDF documents
        logger.info("Querying PDF collection...")
        context = self.vector_store.query_pdf_collection(query)
        logger.info(f"Retrieved {len(context)} context chunks")
        
        if context:
            # If we found relevant PDF context, use it
            for i, ctx in enumerate(context):
                source = ctx["metadata"].get("source", "Unknown")
                pages = ctx["metadata"].get("page_numbers", [])
                logger.info(f"Context chunk {i+1}:")
                logger.info(f"- Source: {source}")
                logger.info(f"- Pages: {pages}")
                logger.info(f"- Content preview: {ctx['content'][:100]}...")
            
            logger.info("Generating response with PDF context...")
            response = self._generate_response(query, context)
            logger.info("Response generated successfully")
            return response
        
        # If no PDF context found or if it's a general knowledge query,
        # use general knowledge
        logger.info("No relevant PDF context found or general knowledge query detected")
        logger.info("Using general knowledge response...")
        return self._generate_general_response(query)
    
    def _generate_response(self, query: str, context: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a response using the retrieved context"""
        logger.info("Preparing context for response generation...")
        context_str = "\n\n".join([f"Context {i+1}:\n{item['content']}" 
                                  for i, item in enumerate(context)])
        
        logger.info("Building prompt with context...")
        if self.language == "es":
            if self.use_cot:
                prompt = f"""Responde a la siguiente consulta en español usando el contexto proporcionado y razonamiento paso a paso.
Primero divide el problema en pasos, luego usa el contexto para resolver cada paso y llegar a la respuesta final.
Si el contexto no contiene suficiente información para responder con precisión, dilo explícitamente.

Contexto:
{context_str}

Consulta: {query}

Pensemos en esto paso a paso:"""
            else:
                prompt = f"""Responde a la siguiente consulta en español usando el contexto proporcionado.
Si el contexto no contiene suficiente información para responder con precisión,
dilo explícitamente.

Contexto:
{context_str}

Consulta: {query}

Respuesta:"""
        else:
            if self.use_cot:
                prompt = f"""Answer the following query using the provided context and chain of thought reasoning.
First break down the problem into steps, then use the context to solve each step and arrive at the final answer.
If the context doesn't contain enough information to answer accurately, say so explicitly.

Context:
{context_str}

Query: {query}

Let's think about this step by step:"""
            else:
                prompt = f"""Answer the following query using the provided context. 
If the context doesn't contain enough information to answer accurately, 
say so explicitly.

Context:
{context_str}

Query: {query}

Answer:"""
        
        logger.info("Generating response using local model...")
        response = self._generate_text(prompt, max_length=1024)
        logger.info("Response generation complete")
        
        return {
            "answer": response,
            "context": context
        }

    def _generate_general_response(self, query: str) -> Dict[str, Any]:
        """Generate a response using general knowledge when no context is available"""
        logger.info("Generating general knowledge response...")
        
        if self.language == "es":
            if self.use_cot:
                prompt = f"""Eres un asistente de IA útil. Si bien no tengo información específica de mi colección de documentos sobre esta consulta, compartiré lo que sé al respecto.

Por favor, responde a la siguiente consulta en español usando razonamiento paso a paso:

Consulta: {query}

Pensemos en esto paso a paso:"""
            else:
                prompt = f"""Eres un asistente de IA útil. Si bien no tengo información específica de mi colección de documentos sobre esta consulta, compartiré lo que sé al respecto.

Consulta: {query}

Respuesta:"""
        else:
            if self.use_cot:
                prompt = f"""You are a helpful AI assistant. While I don't have specific information from my document collection about this query, I'll share what I know about it.

Please answer the following query using chain of thought reasoning:

Query: {query}

Let's think about this step by step:"""
            else:
                prompt = f"""You are a helpful AI assistant. While I don't have specific information from my document collection about this query, I'll share what I know about it.

Query: {query}

Answer:"""
        
        logger.info("Generating response using local model...")
        response = self._generate_text(prompt, max_length=1024)
        logger.info("Response generation complete")
        
        prefix = "No encontré información específica en mis documentos, pero esto es lo que sé al respecto:\n\n" if self.language == "es" else "I didn't find specific information in my documents, but here's what I know about it:\n\n"
        
        return {
            "answer": prefix + response,
            "context": []
        }

def main():
    parser = argparse.ArgumentParser(description="Query documents using local Mistral model")
    parser.add_argument("--query", required=True, help="Query to process")
    parser.add_argument("--store-path", default="embeddings", help="Path to the vector store")
    parser.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.2", help="Model to use")
    parser.add_argument("--quiet", action="store_true", help="Disable verbose logging")
    parser.add_argument("--use-cot", action="store_true", help="Enable Chain of Thought reasoning")
    
    args = parser.parse_args()
    
    # Set logging level based on quiet flag
    if args.quiet:
        logger.setLevel(logging.WARNING)
    else:
        logger.setLevel(logging.INFO)
    
    print("\nInitializing RAG agent...")
    print("=" * 50)
    
    try:
        logger.info(f"Initializing vector store from: {args.store_path}")
        store = VectorStore(persist_directory=args.store_path)
        logger.info("Initializing local RAG agent...")
        agent = LocalRAGAgent(store, model_name=args.model, use_cot=args.use_cot)
        
        print(f"\nProcessing query: {args.query}")
        print("=" * 50)
        
        response = agent.process_query(args.query)
        
        print("\nResponse:")
        print("-" * 50)
        print(response["answer"])
        
        if response.get("context"):
            print("\nSources used:")
            for ctx in response["context"]:
                source = ctx["metadata"].get("source", "Unknown")
                pages = ctx["metadata"].get("page_numbers", [])
                print(f"- {source} (pages: {pages})")
    
    except Exception as e:
        logger.error(f"Error during execution: {str(e)}", exc_info=True)
        print(f"\n✗ Error: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main() 