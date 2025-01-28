from typing import List, Dict, Any, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from pydantic import BaseModel, Field
from store import VectorStore
import argparse
import yaml
import os

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
    def __init__(self, vector_store: VectorStore, model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"):
        """Initialize local RAG agent with vector store and local LLM"""
        self.vector_store = vector_store
        
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
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process a user query using the agentic RAG pipeline"""
        # Analyze the query
        analysis = self._analyze_query(query)
        
        # If query type is unsupported, return early
        if analysis.query_type == "unsupported":
            return {
                "answer": "I apologize, but I don't have the information to answer this query.",
                "reasoning": analysis.reasoning,
                "context": []
            }
        
        # Retrieve relevant context based on query type
        if analysis.query_type == "pdf_documents":
            context = self.vector_store.query_pdf_collection(query)
        else:
            context = self.vector_store.query_general_collection(query)
        
        # Generate response using context
        if context and analysis.requires_context:
            response = self._generate_response(query, context)
        else:
            response = {
                "answer": "I couldn't find relevant information to answer your query.",
                "reasoning": analysis.reasoning,
                "context": []
            }
        
        return response
    
    def _generate_response(self, query: str, context: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a response using the retrieved context"""
        context_str = "\n\n".join([f"Context {i+1}:\n{item['content']}" 
                                  for i, item in enumerate(context)])
        
        prompt = f"""Answer the following query using the provided context. 
If the context doesn't contain enough information to answer accurately, 
say so explicitly.

Context:
{context_str}

Query: {query}

Answer:"""
        
        response = self._generate_text(prompt, max_length=1024)
        
        return {
            "answer": response,
            "context": context
        }

def main():
    parser = argparse.ArgumentParser(description="Query documents using local Mistral model")
    parser.add_argument("--query", required=True, help="Query to process")
    parser.add_argument("--store-path", default="chroma_db", help="Path to the vector store")
    parser.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.2", help="Model to use")
    
    args = parser.parse_args()
    
    print("\nInitializing RAG agent...")
    print("=" * 50)
    
    try:
        store = VectorStore(persist_directory=args.store_path)
        agent = LocalRAGAgent(store, model_name=args.model)
        
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
        print(f"\n✗ Error: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main() 