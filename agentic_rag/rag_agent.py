from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from store import VectorStore
import os
import argparse
from dotenv import load_dotenv

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

class RAGAgent:
    def __init__(self, vector_store: VectorStore, openai_api_key: str, use_cot: bool = False):
        """Initialize RAG agent with vector store and LLM"""
        self.vector_store = vector_store
        self.use_cot = use_cot
        self.llm = ChatOpenAI(
            model="gpt-4-turbo-preview",
            temperature=0,
            api_key=openai_api_key
        )
        self.query_analyzer = self._create_query_analyzer()
        
    def _create_query_analyzer(self):
        """Create a chain for analyzing queries"""
        template = """You are an intelligent agent that analyzes user queries to determine the best source of information.
        
        Analyze the following query and determine:
        1. Whether it should query the PDF documents collection or general knowledge collection
        2. Your reasoning for this decision
        3. Whether the query requires additional context to provide a good answer
        
        Query: {query}
        
        {format_instructions}
        """
        
        prompt = ChatPromptTemplate.from_template(template)
        output_parser = PydanticOutputParser(pydantic_object=QueryAnalysis)
        
        prompt = prompt.partial(format_instructions=output_parser.get_format_instructions())
        
        return {"prompt": prompt, "parser": output_parser}
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process a user query using the agentic RAG pipeline"""
        # Analyze the query
        analysis = self._analyze_query(query)
        
        # If query type is unsupported, use general knowledge
        if analysis.query_type == "unsupported":
            return self._generate_general_response(query)
        
        # Retrieve relevant context based on query type
        if analysis.query_type == "pdf_documents":
            context = self.vector_store.query_pdf_collection(query)
        else:
            context = self.vector_store.query_general_collection(query)
        
        # Generate response using context if available, otherwise use general knowledge
        if context and analysis.requires_context:
            response = self._generate_response(query, context)
        else:
            response = self._generate_general_response(query)
        
        return response
    
    def _analyze_query(self, query: str) -> QueryAnalysis:
        """Analyze the query to determine the best source of information"""
        chain_input = {"query": query}
        result = self.llm.invoke(self.query_analyzer["prompt"].format_messages(**chain_input))
        return self.query_analyzer["parser"].parse(result.content)
    
    def _generate_response(self, query: str, context: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a response using the retrieved context"""
        context_str = "\n\n".join([f"Context {i+1}:\n{item['content']}" 
                                  for i, item in enumerate(context)])
        
        if self.use_cot:
            template = """Answer the following query using the provided context and chain of thought reasoning.
First break down the problem into steps, then use the context to solve each step and arrive at the final answer.
If the context doesn't contain enough information to answer accurately, say so explicitly.

Context:
{context}

Query: {query}

Let's think about this step by step:"""
        else:
            template = """Answer the following query using the provided context. 
If the context doesn't contain enough information to answer accurately, 
say so explicitly.

Context:
{context}

Query: {query}

Answer:"""
        
        prompt = ChatPromptTemplate.from_template(template)
        
        messages = prompt.format_messages(context=context_str, query=query)
        response = self.llm.invoke(messages)
        
        return {
            "answer": response.content,
            "context": context
        }

    def _generate_general_response(self, query: str) -> Dict[str, Any]:
        """Generate a response using general knowledge when no context is available"""
        if self.use_cot:
            template = """You are a helpful AI assistant. While I don't have specific information from my document collection about this query, I'll share what I generally know about it.

Please answer the following query using chain of thought reasoning:

Query: {query}

Let's think about this step by step:"""
        else:
            template = """You are a helpful AI assistant. While I don't have specific information from my document collection about this query, I'll share what I generally know about it.

Query: {query}

Answer:"""
        
        prompt = ChatPromptTemplate.from_template(template)
        messages = prompt.format_messages(query=query)
        response = self.llm.invoke(messages)
        
        return {
            "answer": "I didn't find specific information in my documents, but here's what I know about it:\n\n" + response.content,
            "context": []
        }

def main():
    parser = argparse.ArgumentParser(description="Query documents using OpenAI GPT-4")
    parser.add_argument("--query", required=True, help="Query to process")
    parser.add_argument("--store-path", default="chroma_db", help="Path to the vector store")
    parser.add_argument("--use-cot", action="store_true", help="Enable Chain of Thought reasoning")
    
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    if not os.getenv("OPENAI_API_KEY"):
        print("✗ Error: OPENAI_API_KEY not found in environment variables")
        print("Please create a .env file with your OpenAI API key")
        exit(1)
    
    print("\nInitializing RAG agent...")
    print("=" * 50)
    
    try:
        store = VectorStore(persist_directory=args.store_path)
        agent = RAGAgent(store, openai_api_key=os.getenv("OPENAI_API_KEY"), use_cot=args.use_cot)
        
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