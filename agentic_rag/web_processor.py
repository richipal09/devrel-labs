from pathlib import Path
import json
import argparse
from typing import List, Dict, Any
from trafilatura import fetch_url, extract, extract_metadata
from urllib.parse import urlparse

def is_url(string: str) -> bool:
    """Check if a string is a valid URL"""
    try:
        result = urlparse(string)
        return all([result.scheme, result.netloc])
    except:
        return False

class WebProcessor:
    def __init__(self, chunk_size: int = 500):
        """Initialize web processor with chunk size"""
        self.chunk_size = chunk_size
    
    def _chunk_text(self, text: str) -> List[str]:
        """Split text into chunks of roughly equal size"""
        # Split into sentences (roughly)
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            # Add period back
            sentence = sentence + '.'
            # If adding this sentence would exceed chunk size, save current chunk
            if current_length + len(sentence) > self.chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_length = 0
            
            current_chunk.append(sentence)
            current_length += len(sentence)
        
        # Add any remaining text
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks

    def process_url(self, url: str) -> List[Dict[str, Any]]:
        """Process a URL and return chunks of text with metadata"""
        try:
            # Download and extract content
            downloaded = fetch_url(url)
            if not downloaded:
                raise ValueError(f"Failed to fetch URL: {url}")
            
            # Extract text and metadata
            text = extract(downloaded, include_comments=False, include_tables=False)
            metadata = extract_metadata(downloaded)
            
            if not text:
                raise ValueError(f"No text content extracted from URL: {url}")
            
            # Split into chunks
            text_chunks = self._chunk_text(text)
            
            # Process chunks into a standardized format
            processed_chunks = []
            for i, chunk in enumerate(text_chunks):
                processed_chunk = {
                    "text": chunk,
                    "metadata": {
                        "source": url,
                        "title": metadata.get('title', ''),
                        "author": metadata.get('author', ''),
                        "date": metadata.get('date', ''),
                        "sitename": metadata.get('sitename', ''),
                        "categories": metadata.get('categories', []),
                        "tags": metadata.get('tags', []),
                        "chunk_id": i
                    }
                }
                processed_chunks.append(processed_chunk)
            
            return processed_chunks
        
        except Exception as e:
            raise Exception(f"Error processing URL {url}: {str(e)}")
    
    def process_urls(self, urls: List[str]) -> List[Dict[str, Any]]:
        """Process multiple URLs and return combined chunks"""
        all_chunks = []
        
        for url in urls:
            try:
                chunks = self.process_url(url)
                all_chunks.extend(chunks)
                print(f"✓ Processed {url}")
            except Exception as e:
                print(f"✗ Failed to process {url}: {str(e)}")
        
        return all_chunks

def main():
    parser = argparse.ArgumentParser(description="Process web pages and extract text chunks")
    parser.add_argument("--input", required=True, help="Input URL or file containing URLs (one per line)")
    parser.add_argument("--output", required=True, help="Output JSON file for chunks")
    parser.add_argument("--chunk-size", type=int, default=500, help="Maximum size of text chunks")
    
    args = parser.parse_args()
    processor = WebProcessor(chunk_size=args.chunk_size)
    
    try:
        # Create output directory if it doesn't exist
        output_dir = Path(args.output).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if is_url(args.input):
            print(f"\nProcessing URL: {args.input}")
            print("=" * 50)
            chunks = processor.process_url(args.input)
        else:
            # Read URLs from file
            with open(args.input, 'r', encoding='utf-8') as f:
                urls = [line.strip() for line in f if line.strip()]
            
            print(f"\nProcessing {len(urls)} URLs from: {args.input}")
            print("=" * 50)
            chunks = processor.process_urls(urls)
        
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