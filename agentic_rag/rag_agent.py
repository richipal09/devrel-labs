from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from store import VectorStore
from agents.agent_factory import create_agents
import os
import argparse
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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

class RAGAgent:
    def __init__(self, vector_store: VectorStore, openai_api_key: str, use_cot: bool = False, language: str = "en"):
        """Initialize RAG agent with vector store and LLM"""
        self.vector_store = vector_store
        self.use_cot = use_cot
        self.language = language
        self.llm = ChatOpenAI(
            model="gpt-4-turbo-preview",
            temperature=0,
            api_key=openai_api_key
        )
        self.query_analyzer = self._create_query_analyzer()
        
        # Initialize specialized agents
        self.agents = create_agents(self.llm, vector_store) if use_cot else None
    
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
        logger.info(f"Query analysis: {analysis}")
        
        if self.use_cot:
            return self._process_query_with_cot(query, analysis)
        else:
            return self._process_query_standard(query, analysis)
    
    def _process_query_with_cot(self, query: str, analysis: QueryAnalysis) -> Dict[str, Any]:
        """Process query using Chain of Thought reasoning with multiple agents"""
        logger.info("Processing query with Chain of Thought reasoning")
        
        # Get initial context if needed
        initial_context = []
        if analysis.requires_context and analysis.query_type != "unsupported":
            pdf_context = self.vector_store.query_pdf_collection(query)
            repo_context = self.vector_store.query_repo_collection(query)
            initial_context = pdf_context + repo_context
        
        # Step 1: Planning
        logger.info("Step 1: Planning")
        plan = self.agents["planner"].plan(query, initial_context)
        logger.info(f"Generated plan:\n{plan}")
        
        # Step 2: Research each step (if researcher is available)
        logger.info("Step 2: Research")
        research_results = []
        if self.agents["researcher"] is not None and initial_context:
            for step in plan.split("\n"):
                if not step.strip():
                    continue
                step_research = self.agents["researcher"].research(query, step)
                research_results.append({"step": step, "findings": step_research})
                logger.info(f"Research for step: {step}\nFindings: {step_research}")
        else:
            # If no researcher or no context, use the steps directly
            research_results = [{"step": step, "findings": []} for step in plan.split("\n") if step.strip()]
            logger.info("No research performed (no researcher agent or no context available)")
        
        # Step 3: Reasoning about each step
        logger.info("Step 3: Reasoning")
        reasoning_steps = []
        for result in research_results:
            step_reasoning = self.agents["reasoner"].reason(
                query,
                result["step"],
                result["findings"] if result["findings"] else [{"content": "Using general knowledge", "metadata": {"source": "General Knowledge"}}]
            )
            reasoning_steps.append(step_reasoning)
            logger.info(f"Reasoning for step: {result['step']}\n{step_reasoning}")
        
        # Step 4: Synthesize final answer
        logger.info("Step 4: Synthesis")
        final_answer = self.agents["synthesizer"].synthesize(query, reasoning_steps)
        logger.info(f"Final synthesized answer:\n{final_answer}")
        
        return {
            "answer": final_answer,
            "context": initial_context,
            "reasoning_steps": reasoning_steps
        }
    
    def _process_query_standard(self, query: str, analysis: QueryAnalysis) -> Dict[str, Any]:
        """Process query using standard approach without Chain of Thought"""
        # If query type is unsupported, use general knowledge
        if analysis.query_type == "unsupported":
            return self._generate_general_response(query)
        
        # First try to get context from PDF documents
        pdf_context = self.vector_store.query_pdf_collection(query)
        
        # Then try repository documents
        repo_context = self.vector_store.query_repo_collection(query)
        
        # Combine all context
        all_context = pdf_context + repo_context
        
        # Generate response using context if available, otherwise use general knowledge
        if all_context and analysis.requires_context:
            response = self._generate_response(query, all_context)
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
        template = """You are a helpful AI assistant. While I don't have specific information from my document collection about this query, I'll share what I know about it.

Query: {query}

Answer:"""
        
        prompt = ChatPromptTemplate.from_template(template)
        messages = prompt.format_messages(query=query)
        response = self.llm.invoke(messages)
        
        prefix = "I didn't find specific information in my documents, but here's what I know about it:\n\n"
        
        return {
            "answer": prefix + response.content,
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
        
        if response.get("reasoning_steps"):
            print("\nReasoning Steps:")
            print("-" * 50)
            for i, step in enumerate(response["reasoning_steps"]):
                print(f"\nStep {i+1}:")
                print(step)
        
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