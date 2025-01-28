from pathlib import Path
from typing import List, Dict, Any
import json
import argparse
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker

class PDFProcessor:
    def __init__(self, tokenizer: str = "BAAI/bge-small-en-v1.5"):
        """Initialize PDF processor with Docling components"""
        self.converter = DocumentConverter()
        self.chunker = HybridChunker(tokenizer=tokenizer)
    
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
                processed_chunk = {
                    "text": chunk["text"],
                    "metadata": {
                        "source": str(file_path),
                        "headings": chunk["meta"].get("headings", []),
                        "page_numbers": self._extract_page_numbers(chunk["meta"]),
                    }
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
                processed_chunk = {
                    "text": chunk["text"],
                    "metadata": {
                        "source": url,
                        "headings": chunk["meta"].get("headings", []),
                        "page_numbers": self._extract_page_numbers(chunk["meta"]),
                    }
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
    
    def _extract_page_numbers(self, meta: Dict) -> List[int]:
        """Extract page numbers from chunk metadata"""
        page_numbers = set()
        if "doc_items" in meta:
            for item in meta["doc_items"]:
                if "prov" in item:
                    for prov in item["prov"]:
                        if "page_no" in prov:
                            page_numbers.add(prov["page_no"])
        return sorted(list(page_numbers))

def main():
    parser = argparse.ArgumentParser(description="Process PDF files and extract text chunks")
    parser.add_argument("--input", required=True, 
                       help="Input PDF file, directory, or URL (http/https URLs supported)")
    parser.add_argument("--output", required=True, help="Output JSON file for chunks")
    parser.add_argument("--tokenizer", default="BAAI/bge-small-en-v1.5", help="Tokenizer to use for chunking")
    
    args = parser.parse_args()
    processor = PDFProcessor(tokenizer=args.tokenizer)
    
    try:
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