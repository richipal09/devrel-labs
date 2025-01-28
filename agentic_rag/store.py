from typing import List, Dict, Any
import chromadb
import json
import argparse
from chromadb.config import Settings

class VectorStore:
    def __init__(self, persist_directory: str = "chroma_db"):
        """Initialize vector store with ChromaDB"""
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(allow_reset=True)
        )
        
        # Create or get collections
        self.pdf_collection = self.client.get_or_create_collection(
            name="pdf_documents",
            metadata={"hnsw:space": "cosine"}
        )
        self.general_collection = self.client.get_or_create_collection(
            name="general_knowledge",
            metadata={"hnsw:space": "cosine"}
        )
    
    def _sanitize_metadata(self, metadata: Dict) -> Dict:
        """Sanitize metadata to ensure all values are valid types for ChromaDB"""
        sanitized = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                sanitized[key] = value
            elif isinstance(value, list):
                # Convert list to string representation
                sanitized[key] = str(value)
            elif value is None:
                # Replace None with empty string
                sanitized[key] = ""
            else:
                # Convert any other type to string
                sanitized[key] = str(value)
        return sanitized
    
    def add_pdf_chunks(self, chunks: List[Dict[str, Any]], document_id: str):
        """Add chunks from a PDF document to the vector store"""
        if not chunks:
            return
        
        # Prepare data for ChromaDB
        texts = [chunk["text"] for chunk in chunks]
        metadatas = [self._sanitize_metadata(chunk["metadata"]) for chunk in chunks]
        ids = [f"{document_id}_{i}" for i in range(len(chunks))]
        
        # Add to collection
        self.pdf_collection.add(
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
    
    def add_general_knowledge(self, chunks: List[Dict[str, Any]], source_id: str):
        """Add general knowledge chunks to the vector store"""
        if not chunks:
            return
        
        # Prepare data for ChromaDB
        texts = [chunk["text"] for chunk in chunks]
        metadatas = [self._sanitize_metadata(chunk["metadata"]) for chunk in chunks]
        ids = [f"{source_id}_{i}" for i in range(len(chunks))]
        
        # Add to collection
        self.general_collection.add(
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
    
    def query_pdf_collection(self, query: str, n_results: int = 3) -> List[Dict[str, Any]]:
        """Query the PDF documents collection"""
        results = self.pdf_collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results["documents"][0])):
            result = {
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i]
            }
            formatted_results.append(result)
        
        return formatted_results
    
    def query_general_collection(self, query: str, n_results: int = 3) -> List[Dict[str, Any]]:
        """Query the general knowledge collection"""
        results = self.general_collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results["documents"][0])):
            result = {
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i]
            }
            formatted_results.append(result)
        
        return formatted_results

def main():
    parser = argparse.ArgumentParser(description="Manage vector store")
    parser.add_argument("--add", help="JSON file containing chunks to add")
    parser.add_argument("--query", help="Query to search for")
    parser.add_argument("--store-path", default="chroma_db", help="Path to vector store")
    
    args = parser.parse_args()
    store = VectorStore(persist_directory=args.store_path)
    
    if args.add:
        with open(args.add, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        store.add_pdf_chunks(chunks, document_id=args.add)
        print(f"✓ Added {len(chunks)} chunks to vector store")
    
    if args.query:
        results = store.query_pdf_collection(args.query)
        print("\nResults:")
        print("-" * 50)
        for result in results:
            print(f"Content: {result['content'][:200]}...")
            print(f"Source: {result['metadata'].get('source', 'Unknown')}")
            print(f"Pages: {result['metadata'].get('page_numbers', [])}")
            print("-" * 50)

if __name__ == "__main__":
    main() 