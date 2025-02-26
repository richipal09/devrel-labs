import gradio as gr
import os
from typing import List, Dict, Any
from pathlib import Path
import tempfile
from dotenv import load_dotenv
import yaml

from pdf_processor import PDFProcessor
from web_processor import WebProcessor
from repo_processor import RepoProcessor
from store import VectorStore
from local_rag_agent import LocalRAGAgent
from rag_agent import RAGAgent

# Load environment variables and config
load_dotenv()

def load_config():
    """Load configuration from config.yaml"""
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        return config.get('HUGGING_FACE_HUB_TOKEN')
    except Exception as e:
        print(f"Error loading config: {str(e)}")
        return None

# Initialize components
pdf_processor = PDFProcessor()
web_processor = WebProcessor()
repo_processor = RepoProcessor()
vector_store = VectorStore()

# Initialize agents
hf_token = load_config()
openai_key = os.getenv("OPENAI_API_KEY")

# Initialize agents with use_cot=True to ensure CoT is available
local_agent = LocalRAGAgent(vector_store, use_cot=True) if hf_token else None
openai_agent = RAGAgent(vector_store, openai_api_key=openai_key, use_cot=True) if openai_key else None

def process_pdf(file: tempfile._TemporaryFileWrapper) -> str:
    """Process uploaded PDF file"""
    try:
        chunks, document_id = pdf_processor.process_pdf(file.name)
        vector_store.add_pdf_chunks(chunks, document_id=document_id)
        return f"✓ Successfully processed PDF and added {len(chunks)} chunks to knowledge base (ID: {document_id})"
    except Exception as e:
        return f"✗ Error processing PDF: {str(e)}"

def process_url(url: str) -> str:
    """Process web content from URL"""
    try:
        # Process URL and get chunks
        chunks = web_processor.process_url(url)
        if not chunks:
            return "✗ No content extracted from URL"
            
        # Add chunks to vector store with URL as source ID
        vector_store.add_web_chunks(chunks, source_id=url)
        return f"✓ Successfully processed URL and added {len(chunks)} chunks to knowledge base"
    except Exception as e:
        return f"✗ Error processing URL: {str(e)}"

def process_repo(repo_path: str) -> str:
    """Process repository content"""
    try:
        # Process repository and get chunks
        chunks, document_id = repo_processor.process_repo(repo_path)
        if not chunks:
            return "✗ No content extracted from repository"
            
        # Add chunks to vector store
        vector_store.add_repo_chunks(chunks, document_id=document_id)
        return f"✓ Successfully processed repository and added {len(chunks)} chunks to knowledge base (ID: {document_id})"
    except Exception as e:
        return f"✗ Error processing repository: {str(e)}"

def chat(message: str, history: List[List[str]], agent_type: str, use_cot: bool, language: str, collection: str) -> List[List[str]]:
    """Process chat message using selected agent and collection"""
    try:
        print("\n" + "="*50)
        print(f"New message received: {message}")
        print(f"Agent: {agent_type}, CoT: {use_cot}, Language: {language}, Collection: {collection}")
        print("="*50 + "\n")
        
        # Determine if we should skip analysis based on collection and interface type
        # Skip analysis for General Knowledge or when using standard chat interface (not CoT)
        skip_analysis = collection == "General Knowledge" or not use_cot
        
        # Select appropriate agent and reinitialize with correct settings
        if agent_type == "Local (Mistral)":
            if not hf_token:
                response_text = "Local agent not available. Please check your HuggingFace token configuration."
                print(f"Error: {response_text}")
                return history + [[message, response_text]]
            agent = LocalRAGAgent(vector_store, use_cot=use_cot, collection=collection, skip_analysis=skip_analysis)
        else:
            if not openai_key:
                response_text = "OpenAI agent not available. Please check your OpenAI API key configuration."
                print(f"Error: {response_text}")
                return history + [[message, response_text]]
            agent = RAGAgent(vector_store, openai_api_key=openai_key, use_cot=use_cot, collection=collection, skip_analysis=skip_analysis)
        
        # Convert language selection to language code
        lang_code = "es" if language == "Spanish" else "en"
        agent.language = lang_code
        
        # Process query and get response
        print("Processing query...")
        response = agent.process_query(message)
        print("Query processed successfully")
        
        # Format response with reasoning steps if CoT is enabled
        if use_cot and "reasoning_steps" in response:
            formatted_response = "🤔 Let me think about this step by step:\n\n"
            print("\nChain of Thought Reasoning Steps:")
            print("-" * 50)
            
            # Add each reasoning step with conclusion
            for i, step in enumerate(response["reasoning_steps"], 1):
                step_text = f"Step {i}:\n{step}\n"
                formatted_response += step_text
                print(step_text)
                
                # Add intermediate response to chat history to show progress
                history.append([None, f"🔄 Step {i} Conclusion:\n{step}"])
            
            # Add final answer
            print("\nFinal Answer:")
            print("-" * 50)
            final_answer = "\n🎯 Final Answer:\n" + response["answer"]
            formatted_response += final_answer
            print(final_answer)
            
            # Add sources if available
            if response.get("context"):
                print("\nSources Used:")
                print("-" * 50)
                sources_text = "\n📚 Sources used:\n"
                formatted_response += sources_text
                print(sources_text)
                
                for ctx in response["context"]:
                    source = ctx["metadata"].get("source", "Unknown")
                    if "page_numbers" in ctx["metadata"]:
                        pages = ctx["metadata"].get("page_numbers", [])
                        source_line = f"- {source} (pages: {pages})\n"
                    else:
                        file_path = ctx["metadata"].get("file_path", "Unknown")
                        source_line = f"- {source} (file: {file_path})\n"
                    formatted_response += source_line
                    print(source_line)
            
            # Add final formatted response to history
            history.append([message, formatted_response])
        else:
            formatted_response = response["answer"]
            print("\nStandard Response:")
            print("-" * 50)
            print(formatted_response)
            history.append([message, formatted_response])
        
        print("\n" + "="*50)
        print("Response complete")
        print("="*50 + "\n")
        
        return history
    except Exception as e:
        error_msg = f"Error processing query: {str(e)}"
        print(f"\nError occurred:")
        print("-" * 50)
        print(error_msg)
        print("="*50 + "\n")
        history.append([message, error_msg])
        return history

def create_interface():
    """Create Gradio interface"""
    with gr.Blocks(title="Agentic RAG System", theme=gr.themes.Soft()) as interface:
        gr.Markdown("""
        # 🤖 Agentic RAG System
        
        Upload PDFs, process web content, repositories, and chat with your documents using local or OpenAI models.
        
        > **Note on Performance**: When using the Local (Mistral) model, initial loading can take 1-5 minutes, and each query may take 30-60 seconds to process depending on your hardware. OpenAI queries are typically much faster.
        """)
        
        with gr.Tab("Document Processing"):
            with gr.Row():
                with gr.Column():
                    pdf_file = gr.File(label="Upload PDF")
                    pdf_button = gr.Button("Process PDF")
                    pdf_output = gr.Textbox(label="PDF Processing Output")
                    
                with gr.Column():
                    url_input = gr.Textbox(label="Enter URL")
                    url_button = gr.Button("Process URL")
                    url_output = gr.Textbox(label="URL Processing Output")
                    
                with gr.Column():
                    repo_input = gr.Textbox(label="Enter Repository Path or URL")
                    repo_button = gr.Button("Process Repository")
                    repo_output = gr.Textbox(label="Repository Processing Output")
        
        with gr.Tab("Standard Chat Interface"):
            with gr.Row():
                with gr.Column():
                    standard_agent_dropdown = gr.Dropdown(
                        choices=["Local (Mistral)", "OpenAI"] if openai_key else ["Local (Mistral)"],
                        value="Local (Mistral)",
                        label="Select Agent"
                    )
                with gr.Column():
                    standard_language_dropdown = gr.Dropdown(
                        choices=["English", "Spanish"],
                        value="English",
                        label="Response Language"
                    )
                with gr.Column():
                    standard_collection_dropdown = gr.Dropdown(
                        choices=["PDF Collection", "Repository Collection", "General Knowledge"],
                        value="PDF Collection",
                        label="Knowledge Collection"
                    )
            gr.Markdown("""
            > **Collection Selection**: 
            > - This interface ALWAYS uses the selected collection without performing query analysis.
            > - "PDF Collection": Will ALWAYS search the PDF documents regardless of query type.
            > - "Repository Collection": Will ALWAYS search the repository code regardless of query type.
            > - "General Knowledge": Will ALWAYS use the model's built-in knowledge without searching collections.
            """)
            standard_chatbot = gr.Chatbot(height=400)
            with gr.Row():
                standard_msg = gr.Textbox(label="Your Message", scale=9)
                standard_send = gr.Button("Send", scale=1)
            standard_clear = gr.Button("Clear Chat")

        with gr.Tab("Chain of Thought Chat Interface"):
            with gr.Row():
                with gr.Column():
                    cot_agent_dropdown = gr.Dropdown(
                        choices=["Local (Mistral)", "OpenAI"] if openai_key else ["Local (Mistral)"],
                        value="Local (Mistral)",
                        label="Select Agent"
                    )
                with gr.Column():
                    cot_language_dropdown = gr.Dropdown(
                        choices=["English", "Spanish"],
                        value="English",
                        label="Response Language"
                    )
                with gr.Column():
                    cot_collection_dropdown = gr.Dropdown(
                        choices=["PDF Collection", "Repository Collection", "General Knowledge"],
                        value="PDF Collection",
                        label="Knowledge Collection"
                    )
            gr.Markdown("""
            > **Collection Selection**: 
            > - When a specific collection is selected, the system will ALWAYS use that collection without analysis:
            >   - "PDF Collection": Will ALWAYS search the PDF documents.
            >   - "Repository Collection": Will ALWAYS search the repository code.
            >   - "General Knowledge": Will ALWAYS use the model's built-in knowledge.
            > - This interface shows step-by-step reasoning and may perform query analysis when needed.
            """)
            cot_chatbot = gr.Chatbot(height=400)
            with gr.Row():
                cot_msg = gr.Textbox(label="Your Message", scale=9)
                cot_send = gr.Button("Send", scale=1)
            cot_clear = gr.Button("Clear Chat")
        
        # Event handlers
        pdf_button.click(process_pdf, inputs=[pdf_file], outputs=[pdf_output])
        url_button.click(process_url, inputs=[url_input], outputs=[url_output])
        repo_button.click(process_repo, inputs=[repo_input], outputs=[repo_output])
        
        # Standard chat handlers
        standard_msg.submit(
            chat,
            inputs=[
                standard_msg,
                standard_chatbot,
                standard_agent_dropdown,
                gr.State(False),  # use_cot=False
                standard_language_dropdown,
                standard_collection_dropdown
            ],
            outputs=[standard_chatbot]
        )
        standard_send.click(
            chat,
            inputs=[
                standard_msg,
                standard_chatbot,
                standard_agent_dropdown,
                gr.State(False),  # use_cot=False
                standard_language_dropdown,
                standard_collection_dropdown
            ],
            outputs=[standard_chatbot]
        )
        standard_clear.click(lambda: None, None, standard_chatbot, queue=False)
        
        # CoT chat handlers
        cot_msg.submit(
            chat,
            inputs=[
                cot_msg,
                cot_chatbot,
                cot_agent_dropdown,
                gr.State(True),  # use_cot=True
                cot_language_dropdown,
                cot_collection_dropdown
            ],
            outputs=[cot_chatbot]
        )
        cot_send.click(
            chat,
            inputs=[
                cot_msg,
                cot_chatbot,
                cot_agent_dropdown,
                gr.State(True),  # use_cot=True
                cot_language_dropdown,
                cot_collection_dropdown
            ],
            outputs=[cot_chatbot]
        )
        cot_clear.click(lambda: None, None, cot_chatbot, queue=False)
        
        # Instructions
        gr.Markdown("""
        ## Instructions
        
        1. **Document Processing**:
           - Upload PDFs using the file uploader
           - Process web content by entering URLs
           - Process repositories by entering paths or GitHub URLs
           - All processed content is added to the knowledge base
        
        2. **Standard Chat Interface**:
           - Quick responses without detailed reasoning steps
           - Select your preferred agent (Local Mistral or OpenAI)
           - Choose your preferred response language
           - Select which knowledge collection to query:
             - **PDF Collection**: Always searches PDF documents
             - **Repository Collection**: Always searches code repositories
             - **General Knowledge**: Uses the model's built-in knowledge without searching collections
        
        3. **Chain of Thought Chat Interface**:
           - Detailed responses with step-by-step reasoning
           - See the planning, research, reasoning, and synthesis steps
           - Great for complex queries or when you want to understand the reasoning process
           - May take longer but provides more detailed and thorough answers
           - Same collection selection options as the Standard Chat Interface
        
        4. **Performance Expectations**:
           - **Local (Mistral) model**: Initial loading takes 1-5 minutes, each query takes 30-60 seconds
           - **OpenAI model**: Much faster responses, typically a few seconds per query
           - Chain of Thought reasoning takes longer for both models
        
        Note: OpenAI agent requires an API key in `.env` file
        """)
    
    return interface

def main():
    # Check configuration
    if not hf_token and not openai_key:
        print("⚠️ Warning: Neither HuggingFace token nor OpenAI key found. Please configure at least one.")
    
    # Launch interface
    interface = create_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        inbrowser=True
    )

if __name__ == "__main__":
    main() 