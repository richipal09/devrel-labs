import gradio as gr
import os
from typing import List, Dict, Any
from pathlib import Path
import tempfile
from dotenv import load_dotenv
import yaml

from pdf_processor import PDFProcessor
from web_processor import WebProcessor
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
vector_store = VectorStore()

# Initialize agents
hf_token = load_config()
openai_key = os.getenv("OPENAI_API_KEY")

local_agent = LocalRAGAgent(vector_store) if hf_token else None
openai_agent = RAGAgent(vector_store, openai_api_key=openai_key) if openai_key else None

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

def chat(message: str, history: List[List[str]], agent_type: str, use_cot: bool, language: str) -> List[List[str]]:
    """Process chat message using selected agent"""
    try:
        # Select appropriate agent
        agent = local_agent if agent_type == "Local (Mistral)" else openai_agent
        if not agent:
            return history + [[message, "Agent not available. Please check your configuration."]]
        
        # Convert language selection to language code
        lang_code = "es" if language == "Spanish" else "en"
        
        # Set CoT option and language
        agent.use_cot = use_cot
        agent.language = lang_code
        
        # Process query
        response = agent.process_query(message)
        
        # Return updated history with new message pair
        history.append([message, response["answer"]])
        return history
    except Exception as e:
        history.append([message, f"Error processing query: {str(e)}"])
        return history

def create_interface():
    """Create Gradio interface"""
    with gr.Blocks(title="Agentic RAG System", theme=gr.themes.Soft()) as interface:
        gr.Markdown("""
        # 🤖 Agentic RAG System
        
        Upload PDFs, process web content, and chat with your documents using local or OpenAI models.
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
        
        with gr.Tab("Chat Interface"):
            with gr.Row():
                with gr.Column():
                    agent_dropdown = gr.Dropdown(
                        choices=["Local (Mistral)", "OpenAI"] if openai_key else ["Local (Mistral)"],
                        value="Local (Mistral)",
                        label="Select Agent"
                    )
                    cot_checkbox = gr.Checkbox(label="Enable Chain of Thought Reasoning", value=False)
                with gr.Column():
                    language_dropdown = gr.Dropdown(
                        choices=["English", "Spanish"],
                        value="English",
                        label="Response Language"
                    )
            chatbot = gr.Chatbot(height=400)
            msg = gr.Textbox(label="Your Message")
            clear = gr.Button("Clear Chat")
        
        # Event handlers
        pdf_button.click(process_pdf, inputs=[pdf_file], outputs=[pdf_output])
        url_button.click(process_url, inputs=[url_input], outputs=[url_output])
        msg.submit(
            chat,
            inputs=[
                msg,
                chatbot,
                agent_dropdown,
                cot_checkbox,
                language_dropdown
            ],
            outputs=[chatbot]
        )
        clear.click(lambda: None, None, chatbot, queue=False)
        
        # Instructions
        gr.Markdown("""
        ## Instructions
        
        1. **Document Processing**:
           - Upload PDFs using the file uploader
           - Process web content by entering URLs
           - All processed content is added to the knowledge base
        
        2. **Chat Interface**:
           - Select your preferred agent (Local Mistral or OpenAI)
           - Toggle Chain of Thought reasoning for more detailed responses
           - Choose your preferred response language (English or Spanish)
           - Chat with your documents using natural language
        
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