from pathlib import Path
from typing import List, Dict, Any, Tuple
import json
import argparse
from urllib.parse import urlparse
import warnings
import uuid
from gitingest import ingest

def is_github_url(url: str) -> bool:
    """Check if a string is a valid GitHub URL"""
    try:
        parsed = urlparse(url)
        return parsed.netloc.lower() == "github.com"
    except:
        return False

class RepoProcessor:
    def __init__(self):
        """Initialize repository processor"""
        pass
    
    def _extract_metadata(self, summary: Dict[str, Any], tree: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metadata from repository summary and tree"""
        # Handle case where summary might be a string
        if isinstance(summary, str):
            return {
                "repo_name": "Unknown",
                "description": "",
                "language": "",
                "topics": [],
                "stars": 0,
                "forks": 0,
                "last_updated": "",
                "file_count": len(tree) if tree else 0
            }
        
        return {
            "repo_name": summary.get("name", ""),
            "description": summary.get("description", ""),
            "language": summary.get("language", ""),
            "topics": summary.get("topics", []),
            "stars": summary.get("stars", 0),
            "forks": summary.get("forks", 0),
            "last_updated": summary.get("updated_at", ""),
            "file_count": len(tree) if tree else 0
        }
    
    def process_repo(self, repo_path: str | Path) -> Tuple[List[Dict[str, Any]], str]:
        """Process a repository and return chunks of content with metadata"""
        try:
            # Generate a unique document ID
            document_id = str(uuid.uuid4())
            
            # Check if it's a GitHub URL
            if isinstance(repo_path, str) and is_github_url(repo_path):
                print(f"Processing GitHub repository: {repo_path}")
            else:
                print(f"Processing local repository: {repo_path}")
            
            # Ingest repository
            summary, tree, content = ingest(str(repo_path))
            
            # Calculate token count based on content type
            def estimate_tokens(content: Any) -> int:
                if isinstance(content, dict):
                    # If content is a dictionary of file contents
                    return int(sum(len(str(c).split()) for c in content.values()) * 1.3)
                elif isinstance(content, str):
                    # If content is a single string
                    return int(len(content.split()) * 1.3)
                else:
                    # If content is in another format, return 0
                    return 0
            
            # Print formatted repository information
            if isinstance(summary, dict):
                repo_name = summary.get("name", "Unknown")
                file_count = len(tree) if tree else 0
            else:
                repo_name = str(repo_path).split('/')[-1]
                file_count = len(tree) if tree else 0
            
            token_count = estimate_tokens(content)
            
            print("\nRepository Information:")
            print("-" * 50)
            print(f"📦 Repository: {repo_name}")
            print(f"📄 Files analyzed: {file_count}")
            print(f"🔤 Estimated tokens: {token_count:,}")
            
            # Extract metadata
            metadata = self._extract_metadata(summary, tree)
            
            # Process content into chunks
            processed_chunks = []
            
            if isinstance(content, dict):
                # Handle dictionary of file contents
                for file_path, file_content in content.items():
                    if isinstance(file_content, str):
                        chunk = {
                            "text": file_content,
                            "metadata": {
                                **metadata,
                                "file_path": file_path,
                                "source": str(repo_path),
                                "document_id": document_id
                            }
                        }
                        processed_chunks.append(chunk)
            elif isinstance(content, str):
                # Handle single string content
                chunk = {
                    "text": content,
                    "metadata": {
                        **metadata,
                        "file_path": "repository_content.txt",
                        "source": str(repo_path),
                        "document_id": document_id
                    }
                }
                processed_chunks.append(chunk)
            
            return processed_chunks, document_id
        
        except Exception as e:
            raise Exception(f"Error processing repository {repo_path}: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Process GitHub repositories and extract content")
    parser.add_argument("--input", required=True, 
                       help="Input repository path or GitHub URL")
    parser.add_argument("--output", required=True, help="Output JSON file for chunks")
    
    args = parser.parse_args()
    processor = RepoProcessor()
    
    try:
        # Create output directory if it doesn't exist
        output_dir = Path(args.output).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nProcessing repository: {args.input}")
        print("=" * 50)
        
        chunks, doc_id = processor.process_repo(args.input)
        
        # Save chunks to JSON
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        
        print("\nSummary:")
        print(f"✓ Processed {len(chunks)} chunks")
        print(f"✓ Document ID: {doc_id}")
        print(f"✓ Saved to {args.output}")
        
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main() 