from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
import json
import argparse
from pathlib import Path

class VectorStore:
    def __init__(self, persist_directory: str = "chroma_db"):
        """Initialize ChromaDB client with persistence"""
        self.client = chromadb.PersistentClient(path=persist_directory)
        self._ensure_collections()
    
    def _ensure_collections(self):
        """Ensure required collections exist"""
        self.pdf_collection = self.client.get_or_create_collection(
            name="pdf_documents",
            metadata={"description": "PDF document chunks"}
        )
        self.general_collection = self.client.get_or_create_collection(
            name="general_knowledge",
            metadata={"description": "General knowledge base"}
        )
    
    def add_pdf_chunks(self, chunks: List[Dict[str, Any]], document_id: str):
        """Add processed PDF chunks to the vector store"""
        documents = [chunk["text"] for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]
        ids = [f"{document_id}_{i}" for i in range(len(chunks))]
        
        self.pdf_collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
    
    def query_pdf_collection(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Query the PDF collection"""
        results = self.pdf_collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        return self._format_results(results)
    
    def query_general_collection(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Query the general knowledge collection"""
        results = self.general_collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        return self._format_results(results)
    
    def _format_results(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Format ChromaDB results into a standardized format"""
        formatted_results = []
        
        if not results["documents"]:
            return formatted_results
            
        for i, doc in enumerate(results["documents"][0]):
            result = {
                "content": doc,
                "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                "score": results["distances"][0][i] if "distances" in results else None
            }
            formatted_results.append(result)
        
        return formatted_results

def main():
    parser = argparse.ArgumentParser(description="Manage and query the vector store")
    parser.add_argument("--store-path", default="chroma_db", help="Path to the vector store")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--add", help="JSON file containing chunks to add")
    group.add_argument("--query", help="Query to search for in the vector store")
    parser.add_argument("--n-results", type=int, default=5, help="Number of results to return for queries")
    
    args = parser.parse_args()
    store = VectorStore(persist_directory=args.store_path)
    
    try:
        if args.add:
            # Add documents to store
            print(f"\nAdding documents from: {args.add}")
            print("=" * 50)
            
            with open(args.add, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
            
            document_id = Path(args.add).stem
            store.add_pdf_chunks(chunks, document_id)
            print(f"✓ Added {len(chunks)} chunks to vector store")
            
        else:
            # Query the store
            print(f"\nQuerying vector store: {args.query}")
            print("=" * 50)
            
            results = store.query_pdf_collection(args.query, args.n_results)
            
            if not results:
                print("No results found")
            else:
                print(f"\nFound {len(results)} results:")
                for i, result in enumerate(results, 1):
                    print(f"\n{i}. Score: {result['score']:.4f}")
                    print(f"   Source: {result['metadata'].get('source', 'Unknown')}")
                    print(f"   Content: {result['content'][:200]}...")
    
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main() 