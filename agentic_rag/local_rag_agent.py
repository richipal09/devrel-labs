from typing import List, Dict, Any, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from store import VectorStore
from agents.agent_factory import create_agents
import argparse
import yaml
import os
import logging
import time
import json
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

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

class GGUFModelHandler:
    """Handler for GGUF models using llama-cpp-python"""
    def __init__(self, model_path_or_repo_id: str):
        """Initialize GGUF model handler
        
        Args:
            model_path_or_repo_id: Local path to GGUF model or HuggingFace repo ID
        """
        self.model_path_or_repo_id = model_path_or_repo_id
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load GGUF model using llama-cpp-python"""
        try:
            from llama_cpp import Llama
            
            # Check if model_path is a local file or HuggingFace repo ID
            if os.path.exists(self.model_path_or_repo_id):
                model_path = self.model_path_or_repo_id
            else:
                # Download from HuggingFace
                from huggingface_hub import hf_hub_download
                
                # Try to load HuggingFace token from config
                try:
                    with open('config.yaml', 'r') as f:
                        config = yaml.safe_load(f)
                    token = config.get('HUGGING_FACE_HUB_TOKEN')
                except Exception:
                    token = None
                
                # Extract repo_id and filename
                parts = self.model_path_or_repo_id.split('/')
                if len(parts) < 2:
                    raise ValueError(f"Invalid HuggingFace repo ID: {self.model_path_or_repo_id}")
                
                repo_id = '/'.join(parts[:2])
                
                # Find the GGUF file in the repo
                from huggingface_hub import list_repo_files
                files = list_repo_files(repo_id, token=token)
                gguf_files = [f for f in files if f.endswith('.gguf')]
                
                if not gguf_files:
                    raise ValueError(f"No GGUF files found in repo: {repo_id}")
                
                # Use the first GGUF file or try to find a specific one if specified
                if len(parts) > 2:
                    # Try to find a specific file if specified in the path
                    specified_file = '/'.join(parts[2:])
                    matching_files = [f for f in gguf_files if specified_file in f]
                    if matching_files:
                        filename = matching_files[0]
                    else:
                        filename = gguf_files[0]
                else:
                    filename = gguf_files[0]
                
                print(f"Downloading GGUF model: {filename} from {repo_id}")
                model_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    token=token
                )
            
            # Determine optimal n_gpu_layers based on available VRAM
            n_gpu_layers = 0
            if torch.cuda.is_available():
                # Get available VRAM in GB
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                
                # Simple heuristic for n_gpu_layers based on VRAM
                if vram_gb > 24:
                    n_gpu_layers = -1  # Use all layers
                elif vram_gb > 16:
                    n_gpu_layers = 32
                elif vram_gb > 8:
                    n_gpu_layers = 24
                elif vram_gb > 4:
                    n_gpu_layers = 16
                else:
                    n_gpu_layers = 8
                
                print(f"CUDA available with {vram_gb:.1f}GB VRAM. Using {n_gpu_layers} GPU layers.")
            else:
                print("CUDA not available. Using CPU only.")
            
            # Load the model
            self.model = Llama(
                model_path=model_path,
                n_ctx=4096,  # Context window size
                n_gpu_layers=n_gpu_layers,
                verbose=False
            )
            
            print(f"✓ GGUF model loaded successfully: {os.path.basename(model_path)}")
            
        except ImportError as e:
            raise ImportError(f"Failed to import llama_cpp. Please install with: pip install llama-cpp-python. Error: {str(e)}")
        except Exception as e:
            raise Exception(f"Failed to load GGUF model: {str(e)}")
    
    def __call__(self, prompt, max_new_tokens=512, temperature=0.1, top_p=0.95, **kwargs):
        """Generate text using the GGUF model"""
        if not self.model:
            raise ValueError("Model not loaded")
        
        # Generate text
        result = self.model(
            prompt,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            echo=False
        )
        
        # Format result to match transformers pipeline output
        formatted_result = [{
            "generated_text": result["choices"][0]["text"]
        }]
        
        return formatted_result

class LocalRAGAgent:
    def __init__(self, vector_store: VectorStore, model_name: str = "mistralai/Mistral-7B-Instruct-v0.2", 
                 use_cot: bool = False, collection: str = None, skip_analysis: bool = False,
                 quantization: str = None):
        """Initialize local RAG agent with vector store and local LLM
        
        Args:
            vector_store: Vector store for retrieving context
            model_name: HuggingFace model name/path or GGUF model path/repo
            use_cot: Whether to use Chain of Thought reasoning
            collection: Collection to search in (PDF, Repository, or General Knowledge)
            skip_analysis: Whether to skip query analysis (kept for backward compatibility)
            quantization: Quantization method to use (None, '4bit', '8bit')
        """
        self.vector_store = vector_store
        self.use_cot = use_cot
        self.collection = collection
        self.quantization = quantization
        self.model_name = model_name
        # skip_analysis parameter kept for backward compatibility but no longer used
        
        # Check if this is a GGUF model
        self.is_gguf = model_name.endswith('.gguf') or 'GGUF' in model_name
        
        if self.is_gguf:
            # Load GGUF model
            print("\nLoading GGUF model...")
            print(f"Model: {model_name}")
            print("Note: Initial loading and inference can take 1-5 minutes depending on your hardware.")
            
            # Initialize GGUF model handler
            self.gguf_handler = GGUFModelHandler(model_name)
            
            # Create pipeline-like interface
            self.pipeline = self.gguf_handler
            
        else:
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
            print(f"Model: {model_name}")
            if quantization:
                print(f"Quantization: {quantization}")
            print("Note: Initial loading and inference can take 1-5 minutes depending on your hardware.")
            print("Subsequent queries will be faster but may still take 30-60 seconds per response.")
            
            # Check if CUDA is available and set appropriate dtype
            if torch.cuda.is_available():
                print("CUDA is available. Using GPU acceleration.")
                dtype = torch.float16
            else:
                print("CUDA is not available. Using CPU only (this will be slow).")
                dtype = torch.float32
            
            # Set up model loading parameters
            model_kwargs = {
                "torch_dtype": dtype,
                "device_map": "auto",
                "token": token,
                "low_cpu_mem_usage": True,
                "offload_folder": "offload"
            }
            
            # Apply quantization if specified
            if quantization == '4bit':
                try:
                    from transformers import BitsAndBytesConfig
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4"
                    )
                    model_kwargs["quantization_config"] = quantization_config
                    print("Using 4-bit quantization with bitsandbytes")
                except ImportError:
                    print("Warning: bitsandbytes not installed. Falling back to standard loading.")
                    print("To use 4-bit quantization, install bitsandbytes: pip install bitsandbytes")
            elif quantization == '8bit':
                try:
                    from transformers import BitsAndBytesConfig
                    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                    model_kwargs["quantization_config"] = quantization_config
                    print("Using 8-bit quantization with bitsandbytes")
                except ImportError:
                    print("Warning: bitsandbytes not installed. Falling back to standard loading.")
                    print("To use 8-bit quantization, install bitsandbytes: pip install bitsandbytes")
            
            # Load model with appropriate settings
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **model_kwargs
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
        logger.info(f"Processing query with collection: {self.collection}")
        
        # Process based on collection type and CoT setting
        if self.collection == "General Knowledge":
            # For General Knowledge, directly use general response
            if self.use_cot:
                return self._process_query_with_cot(query)
            else:
                return self._generate_general_response(query)
        else:
            # For PDF or Repository collections, use context-based processing
            if self.use_cot:
                return self._process_query_with_cot(query)
            else:
                return self._process_query_standard(query)
    
    def _process_query_with_cot(self, query: str) -> Dict[str, Any]:
        """Process query using Chain of Thought reasoning with multiple agents"""
        logger.info("Processing query with Chain of Thought reasoning")
        
        # Get initial context based on selected collection
        initial_context = []
        if self.collection == "PDF Collection":
            logger.info(f"Retrieving context from PDF Collection for query: '{query}'")
            pdf_context = self.vector_store.query_pdf_collection(query)
            initial_context.extend(pdf_context)
            logger.info(f"Retrieved {len(pdf_context)} chunks from PDF Collection")
            # Log each chunk with citation number but not full content
            for i, chunk in enumerate(pdf_context):
                source = chunk["metadata"].get("source", "Unknown")
                pages = chunk["metadata"].get("page_numbers", [])
                logger.info(f"Source [{i+1}]: {source} (pages: {pages})")
                # Only log content preview at debug level
                content_preview = chunk["content"][:150] + "..." if len(chunk["content"]) > 150 else chunk["content"]
                logger.debug(f"Content preview for source [{i+1}]: {content_preview}")
        elif self.collection == "Repository Collection":
            logger.info(f"Retrieving context from Repository Collection for query: '{query}'")
            repo_context = self.vector_store.query_repo_collection(query)
            initial_context.extend(repo_context)
            logger.info(f"Retrieved {len(repo_context)} chunks from Repository Collection")
            # Log each chunk with citation number but not full content
            for i, chunk in enumerate(repo_context):
                source = chunk["metadata"].get("source", "Unknown")
                file_path = chunk["metadata"].get("file_path", "Unknown")
                logger.info(f"Source [{i+1}]: {source} (file: {file_path})")
                # Only log content preview at debug level
                content_preview = chunk["content"][:150] + "..." if len(chunk["content"]) > 150 else chunk["content"]
                logger.debug(f"Content preview for source [{i+1}]: {content_preview}")
        # For General Knowledge, no context is needed
        else:
            logger.info("Using General Knowledge collection, no context retrieval needed")
        
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
                    # Log which sources were used for this step
                    source_indices = [initial_context.index(finding) + 1 for finding in step_research if finding in initial_context]
                    logger.info(f"Research for step: {step}\nUsing sources: {source_indices}")
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
    
    def _process_query_standard(self, query: str) -> Dict[str, Any]:
        """Process query using standard approach without Chain of Thought"""
        # Initialize context variables
        pdf_context = []
        repo_context = []
        
        # Get context based on selected collection
        if self.collection == "PDF Collection":
            logger.info(f"Retrieving context from PDF Collection for query: '{query}'")
            pdf_context = self.vector_store.query_pdf_collection(query)
            logger.info(f"Retrieved {len(pdf_context)} chunks from PDF Collection")
            # Log each chunk with citation number but not full content
            for i, chunk in enumerate(pdf_context):
                source = chunk["metadata"].get("source", "Unknown")
                pages = chunk["metadata"].get("page_numbers", [])
                logger.info(f"Source [{i+1}]: {source} (pages: {pages})")
                # Only log content preview at debug level
                content_preview = chunk["content"][:150] + "..." if len(chunk["content"]) > 150 else chunk["content"]
                logger.debug(f"Content preview for source [{i+1}]: {content_preview}")
        elif self.collection == "Repository Collection":
            logger.info(f"Retrieving context from Repository Collection for query: '{query}'")
            repo_context = self.vector_store.query_repo_collection(query)
            logger.info(f"Retrieved {len(repo_context)} chunks from Repository Collection")
            # Log each chunk with citation number but not full content
            for i, chunk in enumerate(repo_context):
                source = chunk["metadata"].get("source", "Unknown")
                file_path = chunk["metadata"].get("file_path", "Unknown")
                logger.info(f"Source [{i+1}]: {source} (file: {file_path})")
                # Only log content preview at debug level
                content_preview = chunk["content"][:150] + "..." if len(chunk["content"]) > 150 else chunk["content"]
                logger.debug(f"Content preview for source [{i+1}]: {content_preview}")
        
        # Combine all context
        all_context = pdf_context + repo_context
        
        # Generate response using context if available, otherwise use general knowledge
        if all_context:
            logger.info(f"Generating response using {len(all_context)} context chunks")
            response = self._generate_response(query, all_context)
        else:
            logger.info("No context found, using general knowledge")
            response = self._generate_general_response(query)
        
        return response
    
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
    parser.add_argument("--verbose", action="store_true", help="Show full content of sources")
    
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
            print("-" * 50)
            
            # Print concise list of sources
            for i, ctx in enumerate(response["context"]):
                source = ctx["metadata"].get("source", "Unknown")
                if "page_numbers" in ctx["metadata"]:
                    pages = ctx["metadata"].get("page_numbers", [])
                    print(f"[{i+1}] {source} (pages: {pages})")
                else:
                    file_path = ctx["metadata"].get("file_path", "Unknown")
                    print(f"[{i+1}] {source} (file: {file_path})")
                
                # Only print content if verbose flag is set
                if args.verbose:
                    content_preview = ctx["content"][:300] + "..." if len(ctx["content"]) > 300 else ctx["content"]
                    print(f"    Content: {content_preview}\n")
    
    except Exception as e:
        logger.error(f"Error during execution: {str(e)}", exc_info=True)
        print(f"\n✗ Error: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main() 