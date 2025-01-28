from pathlib import Path
from typing import List, Dict, Any
import json
import argparse
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from urllib.parse import urlparse

def is_url(string: str) -> bool:
    """Check if a string is a valid URL"""
    try:
        result = urlparse(string)
        return all([result.scheme, result.netloc])
    except:
        return False

class PDFProcessor:
    def __init__(self, tokenizer: str = "BAAI/bge-small-en-v1.5"):
        """Initialize PDF processor with Docling components"""
        self.converter = DocumentConverter()
        self.chunker = HybridChunker(tokenizer=tokenizer, max_chunk_size=384)  # Reduced chunk size
    
    def _extract_metadata(self, meta: Any) -> Dict[str, Any]:
        """Safely extract metadata from various object types"""
        try:
            if hasattr(meta, '__dict__'):
                # If it's an object with attributes
                return {
                    "headings": getattr(meta, "headings", []),
                    "page_numbers": self._extract_page_numbers(meta)
                }
            elif isinstance(meta, dict):
                # If it's a dictionary
                return {
                    "headings": meta.get("headings", []),
                    "page_numbers": self._extract_page_numbers(meta)
                }
            else:
                # Default empty metadata
                return {
                    "headings": [],
                    "page_numbers": []
                }
        except Exception as e:
            print(f"Warning: Error extracting metadata: {str(e)}")
            return {
                "headings": [],
                "page_numbers": []
            }
    
    def process_pdf(self, file_path: str | Path) -> List[Dict[str, Any]]:
        """Process a PDF file and return chunks of text with metadata"""
        try:
            # Convert PDF using Docling
            conv_result = self.converter.convert(file_path)
            if not conv_result or not conv_result.document:
                raise ValueError(f"Failed to convert PDF: {file_path}")
            
            # Chunk the document
            chunks = list(self.chunker.chunk(conv_result.document))
            
            # Process chunks into a standardized format
            processed_chunks = []
            for chunk in chunks:
                # Handle both dictionary and DocChunk objects
                text = chunk.text if hasattr(chunk, 'text') else chunk.get('text', '')
                meta = chunk.meta if hasattr(chunk, 'meta') else chunk.get('meta', {})
                
                metadata = self._extract_metadata(meta)
                metadata["source"] = str(file_path)
                
                processed_chunk = {
                    "text": text,
                    "metadata": metadata
                }
                processed_chunks.append(processed_chunk)
            
            return processed_chunks
        
        except Exception as e:
            raise Exception(f"Error processing PDF {file_path}: {str(e)}")

    def process_pdf_url(self, url: str) -> List[Dict[str, Any]]:
        """Process a PDF file from a URL and return chunks of text with metadata"""
        try:
            # Convert PDF using Docling's built-in URL support
            conv_result = self.converter.convert(url)
            if not conv_result or not conv_result.document:
                raise ValueError(f"Failed to convert PDF from URL: {url}")
            
            # Chunk the document
            chunks = list(self.chunker.chunk(conv_result.document))
            
            # Process chunks into a standardized format
            processed_chunks = []
            for chunk in chunks:
                # Handle both dictionary and DocChunk objects
                text = chunk.text if hasattr(chunk, 'text') else chunk.get('text', '')
                meta = chunk.meta if hasattr(chunk, 'meta') else chunk.get('meta', {})
                
                metadata = self._extract_metadata(meta)
                metadata["source"] = url
                
                processed_chunk = {
                    "text": text,
                    "metadata": metadata
                }
                processed_chunks.append(processed_chunk)
            
            return processed_chunks
        
        except Exception as e:
            raise Exception(f"Error processing PDF from URL {url}: {str(e)}")
    
    def process_directory(self, directory: str | Path) -> List[Dict[str, Any]]:
        """Process all PDF files in a directory"""
        directory = Path(directory)
        all_chunks = []
        
        for pdf_file in directory.glob("**/*.pdf"):
            try:
                chunks = self.process_pdf(pdf_file)
                all_chunks.extend(chunks)
                print(f"✓ Processed {pdf_file}")
            except Exception as e:
                print(f"✗ Failed to process {pdf_file}: {str(e)}")
        
        return all_chunks
    
    def _extract_page_numbers(self, meta: Any) -> List[int]:
        """Extract page numbers from chunk metadata"""
        page_numbers = set()
        try:
            if hasattr(meta, 'doc_items'):
                items = meta.doc_items
            elif isinstance(meta, dict) and 'doc_items' in meta:
                items = meta['doc_items']
            else:
                return []
            
            for item in items:
                if hasattr(item, 'prov'):
                    provs = item.prov
                elif isinstance(item, dict) and 'prov' in item:
                    provs = item['prov']
                else:
                    continue
                
                for prov in provs:
                    if hasattr(prov, 'page_no'):
                        page_numbers.add(prov.page_no)
                    elif isinstance(prov, dict) and 'page_no' in prov:
                        page_numbers.add(prov['page_no'])
            
            return sorted(list(page_numbers))
        except Exception as e:
            print(f"Warning: Error extracting page numbers: {str(e)}")
            return []

def main():
    parser = argparse.ArgumentParser(description="Process PDF files and extract text chunks")
    parser.add_argument("--input", required=True, 
                       help="Input PDF file, directory, or URL (http/https URLs supported)")
    parser.add_argument("--output", required=True, help="Output JSON file for chunks")
    parser.add_argument("--tokenizer", default="BAAI/bge-small-en-v1.5", help="Tokenizer to use for chunking")
    
    args = parser.parse_args()
    processor = PDFProcessor(tokenizer=args.tokenizer)
    
    try:
        # Create output directory if it doesn't exist
        output_dir = Path(args.output).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if is_url(args.input):
            print(f"\nProcessing PDF from URL: {args.input}")
            print("=" * 50)
            chunks = processor.process_pdf_url(args.input)
        elif Path(args.input).is_dir():
            print(f"\nProcessing directory: {args.input}")
            print("=" * 50)
            chunks = processor.process_directory(args.input)
        else:
            print(f"\nProcessing file: {args.input}")
            print("=" * 50)
            chunks = processor.process_pdf(args.input)
        
        # Save chunks to JSON
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        
        print("\nSummary:")
        print(f"✓ Processed {len(chunks)} chunks")
        print(f"✓ Saved to {args.output}")
        
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main() 