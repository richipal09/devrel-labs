from typing import List, Dict, Any, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from pydantic import BaseModel, Field
from store import VectorStore
from agents.agent_factory import create_agents
import argparse
import yaml
import os
import logging
import time

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
        description="Type of query: 'pdf_documents', 'repository_documents', 'general_knowledge', or 'unsupported'"
    )
    reasoning: str = Field(
        description="Reasoning behind the query type selection"
    )
    requires_context: bool = Field(
        description="Whether the query requires additional context to answer"
    )

class LocalLLM:
    """Wrapper for local LLM to match LangChain's ChatOpenAI interface"""
    def __init__(self, pipeline):
        self.pipeline = pipeline
    
    def invoke(self, messages):
        # Convert messages to a single prompt
        prompt = "\n".join([msg.content for msg in messages])
        result = self.pipeline(
            prompt,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.1,
            top_p=0.95,
            return_full_text=False
        )[0]["generated_text"]
        
        # Create a response object with content attribute
        class Response:
            def __init__(self, content):
                self.content = content
        
        return Response(result.strip())

class LocalRAGAgent:
    def __init__(self, vector_store: VectorStore, model_name: str = "mistralai/Mistral-7B-Instruct-v0.2", use_cot: bool = False, collection: str = None, skip_analysis: bool = False):
        """Initialize local RAG agent with vector store and local LLM"""
        self.vector_store = vector_store
        self.use_cot = use_cot
        self.collection = collection
        self.skip_analysis = skip_analysis
        
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
        print("Note: Initial loading and inference with Mistral-7B can take 1-5 minutes depending on your hardware.")
        print("Subsequent queries will be faster but may still take 30-60 seconds per response.")
        
        # Check if CUDA is available and set appropriate dtype
        if torch.cuda.is_available():
            print("CUDA is available. Using GPU acceleration.")
            dtype = torch.float16
        else:
            print("CUDA is not available. Using CPU only (this will be slow).")
            dtype = torch.float32
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto",
            token=token,
            # Add optimization flags
            low_cpu_mem_usage=True,
            offload_folder="offload"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
        
        # Create text generation pipeline with optimized settings
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
        
        # Create LLM wrapper
        self.llm = LocalLLM(self.pipeline)
        
        # Initialize specialized agents if CoT is enabled
        self.agents = create_agents(self.llm, vector_store) if use_cot else None
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process a user query using the agentic RAG pipeline"""
        # Skip analysis if explicitly requested, if General Knowledge is selected, or if not using CoT
        if self.skip_analysis or self.collection == "General Knowledge" or not self.use_cot:
            logger.info("Skipping query analysis (explicitly requested, General Knowledge selected, or standard interface)")
            
            # Create a dummy analysis object for logging consistency
            if self.collection == "PDF Collection":
                analysis = QueryAnalysis(
                    query_type="pdf_documents",
                    reasoning="Using PDF collection as explicitly selected by user",
                    requires_context=True
                )
            elif self.collection == "Repository Collection":
                analysis = QueryAnalysis(
                    query_type="pdf_documents",  # We still use pdf_documents type but will query repo collection
                    reasoning="Using Repository collection as explicitly selected by user",
                    requires_context=True
                )
            else:  # General Knowledge or no collection specified
                analysis = QueryAnalysis(
                    query_type="general_knowledge",
                    reasoning="Using General Knowledge as explicitly selected by user",
                    requires_context=False
                )
            
            logger.info(f"Query analysis (skipped): {analysis}")
            
            # For General Knowledge, directly use general response
            if self.collection == "General Knowledge":
                if self.use_cot:
                    return self._process_query_with_cot(query, analysis)
                else:
                    return self._generate_general_response(query)
            
            # For other collections with skip_analysis, use standard processing with context
            if self.use_cot:
                return self._process_query_with_cot(query, analysis)
            else:
                return self._process_query_standard(query, analysis)
        
        # For cases where analysis is required (only in CoT with no explicit collection)
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
        if analysis.requires_context or self.collection in ["PDF Collection", "Repository Collection"]:
            if self.collection == "PDF Collection" or (not self.collection and analysis.query_type == "pdf_documents"):
                pdf_context = self.vector_store.query_pdf_collection(query)
                initial_context.extend(pdf_context)
            
            if self.collection == "Repository Collection" or (not self.collection and analysis.query_type == "repository_documents"):
                repo_context = self.vector_store.query_repo_collection(query)
                initial_context.extend(repo_context)
            
            # If no specific collection type is identified but context is required, query both
            if not self.collection and analysis.requires_context and analysis.query_type not in ["pdf_documents", "repository_documents", "general_knowledge"]:
                if not initial_context:  # Only if we haven't already added context
                    pdf_context = self.vector_store.query_pdf_collection(query)
                    repo_context = self.vector_store.query_repo_collection(query)
                    initial_context.extend(pdf_context)
                    initial_context.extend(repo_context)
        
        try:
            # Step 1: Planning
            logger.info("Step 1: Planning")
            if not self.agents or "planner" not in self.agents:
                logger.warning("No planner agent available, using direct response")
                return self._generate_general_response(query)
            
            plan = self.agents["planner"].plan(query, initial_context)
            logger.info(f"Generated plan:\n{plan}")
            
            # Step 2: Research each step (if researcher is available)
            logger.info("Step 2: Research")
            research_results = []
            if self.agents.get("researcher") is not None and initial_context:
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
            if not self.agents.get("reasoner"):
                logger.warning("No reasoner agent available, using direct response")
                return self._generate_general_response(query)
            
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
            if not self.agents.get("synthesizer"):
                logger.warning("No synthesizer agent available, using direct response")
                return self._generate_general_response(query)
            
            final_answer = self.agents["synthesizer"].synthesize(query, reasoning_steps)
            logger.info(f"Final synthesized answer:\n{final_answer}")
            
            return {
                "answer": final_answer,
                "context": initial_context,
                "reasoning_steps": reasoning_steps
            }
        except Exception as e:
            logger.error(f"Error in CoT processing: {str(e)}")
            logger.info("Falling back to general response")
            return self._generate_general_response(query)
    
    def _process_query_standard(self, query: str, analysis: QueryAnalysis) -> Dict[str, Any]:
        """Process query using standard approach without Chain of Thought"""
        # If query type is unsupported and not in a specific collection, use general knowledge
        if analysis.query_type == "unsupported" and not self.collection:
            return self._generate_general_response(query)
        
        # Initialize context variables
        pdf_context = []
        repo_context = []
        
        # In standard interface (not CoT), always use the selected collection
        # regardless of analysis.requires_context
        if self.collection == "PDF Collection":
            pdf_context = self.vector_store.query_pdf_collection(query)
        elif self.collection == "Repository Collection":
            repo_context = self.vector_store.query_repo_collection(query)
        # For CoT with no explicit collection, use analysis to determine
        elif not self.collection and analysis.requires_context:
            if analysis.query_type == "pdf_documents":
                pdf_context = self.vector_store.query_pdf_collection(query)
            elif analysis.query_type == "repository_documents":
                repo_context = self.vector_store.query_repo_collection(query)
            else:
                # For other types, query both collections
                pdf_context = self.vector_store.query_pdf_collection(query)
                repo_context = self.vector_store.query_repo_collection(query)
        
        # Combine all context
        all_context = pdf_context + repo_context
        
        # Generate response using context if available, otherwise use general knowledge
        if all_context:
            response = self._generate_response(query, all_context)
        else:
            response = self._generate_general_response(query)
        
        return response
    
    def _analyze_query(self, query: str) -> QueryAnalysis:
        """Analyze the query to determine the best source of information"""
        # If collection is explicitly set, override the analysis
        if self.collection == "PDF Collection":
            return QueryAnalysis(
                query_type="pdf_documents",
                reasoning="Using PDF collection as explicitly selected by user",
                requires_context=True
            )
        elif self.collection == "Repository Collection":
            return QueryAnalysis(
                query_type="repository_documents",  # Changed from pdf_documents to repository_documents
                reasoning="Using Repository collection as explicitly selected by user",
                requires_context=True
            )
        elif self.collection == "General Knowledge":
            return QueryAnalysis(
                query_type="general_knowledge",
                reasoning="Using General Knowledge as explicitly selected by user",
                requires_context=False
            )
        
        # If no collection is explicitly set, perform normal analysis
        prompt = f"""You are an intelligent agent that analyzes user queries to determine the best source of information.

    Analyze the following query and determine:
    1. Whether it should query the PDF documents collection, repository code collection, or general knowledge collection
    2. Your reasoning for this decision
    3. Whether the query requires additional context to provide a good answer

    Query: {query}

    Provide your response in the following JSON format:
    {{
        "query_type": "pdf_documents OR repository_documents OR general_knowledge OR unsupported",
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
    
    def _generate_text(self, prompt: str, max_length: int = 512) -> str:
        """Generate text using the local model"""
        # Log start time for performance monitoring
        start_time = time.time()
        
        result = self.pipeline(
            prompt,
            max_new_tokens=max_length,
            do_sample=True,
            temperature=0.1,
            top_p=0.95,
            return_full_text=False
        )[0]["generated_text"]
        
        # Log completion time
        elapsed_time = time.time() - start_time
        logger.info(f"Text generation completed in {elapsed_time:.2f} seconds")
        
        return result.strip()
    
    def _generate_response(self, query: str, context: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a response using the retrieved context"""
        context_str = "\n\n".join([f"Context {i+1}:\n{item['content']}" 
                                  for i, item in enumerate(context)])
        
        template = """Answer the following query using the provided context. 
Respond as if you are knowledgeable about the topic and incorporate the context naturally.
Do not mention limitations in the context or that you couldn't find specific information.

Context:
{context}

Query: {query}

Answer:"""
        
        prompt = template.format(context=context_str, query=query)
        response = self._generate_text(prompt)
        
        return {
            "answer": response,
            "context": context
        }

    def _generate_general_response(self, query: str) -> Dict[str, Any]:
        """Generate a response using general knowledge when no context is available"""
        template = """You are a helpful AI assistant. Answer the following query using your general knowledge.

Query: {query}

Answer:"""
        
        prompt = template.format(query=query)
        response = self._generate_text(prompt)
        
        return {
            "answer": response,
            "context": []
        }

def main():
    parser = argparse.ArgumentParser(description="Query documents using local Mistral model")
    parser.add_argument("--query", required=True, help="Query to process")
    parser.add_argument("--store-path", default="embeddings", help="Path to the vector store")
    parser.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.2", help="Model to use")
    parser.add_argument("--quiet", action="store_true", help="Disable verbose logging")
    parser.add_argument("--use-cot", action="store_true", help="Enable Chain of Thought reasoning")
    parser.add_argument("--collection", choices=["PDF Collection", "Repository Collection", "General Knowledge"], 
                        help="Specify which collection to query")
    parser.add_argument("--skip-analysis", action="store_true", help="Skip query analysis step")
    
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
        agent = LocalRAGAgent(
            store, 
            model_name=args.model, 
            use_cot=args.use_cot, 
            collection=args.collection,
            skip_analysis=args.skip_analysis
        )
        
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
        logger.error(f"Error during execution: {str(e)}", exc_info=True)
        print(f"\n✗ Error: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main() 