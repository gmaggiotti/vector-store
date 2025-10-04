from pinecone import Pinecone
import json
from typing import List, Dict, Any, Optional
from .vector_store import VectorStore


class PineconeStore(VectorStore):
    """Pinecone implementation of the VectorStore interface."""
    
    def __init__(
        self, 
        api_key: str = None, 
        index_name: str = "my-index",
        key_file: str = "./pinecone_key.json",
        embedding_model: str = "multilingual-e5-large"
    ):
        """Initialize Pinecone store.
        
        Args:
            api_key: Pinecone API key (if None, will load from key_file)
            index_name: Name of the Pinecone index
            key_file: Path to JSON file containing API key
            embedding_model: Model to use for embeddings
        """
        self.index_name = index_name
        self.embedding_model = embedding_model
        
        if api_key is None:
            api_key = self._load_key(key_file)
        
        self.pc = Pinecone(api_key=api_key)
        self.index = None
        
        # Initialize index if it exists
        if self.pc.has_index(index_name):
            self.index = self.pc.Index(index_name)
        else:
            print(f"Warning: Index '{index_name}' does not exist. Call create_index() to create it.")
    
    def _load_key(self, filename: str) -> str:
        """Load API key from JSON file."""
        try:
            with open(filename, 'r') as file:
                data = json.load(file)
            return data['pinecone_api_key']
        except Exception as e:
            raise ValueError(f"Error loading API key from {filename}: {e}")
    
    def create_index(
        self, 
        cloud: str = "aws", 
        region: str = "us-east-1",
        model: str = "llama-text-embed-v2"
    ) -> None:
        """Create a new Pinecone index."""
        try:
            if not self.pc.has_index(self.index_name):
                self.pc.create_index_for_model(
                    name=self.index_name,
                    cloud=cloud,
                    region=region,
                    embed={
                        "model": model,
                        "field_map": {"text": "chunk_text"}
                    }
                )
                print(f"Created index: {self.index_name}")
            
            self.index = self.pc.Index(self.index_name)
            
        except Exception as e:
            print(f"Error creating Pinecone index: {e}")
            raise
    
    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for texts using Pinecone inference API."""
        try:
            results = self.pc.inference.embed(
                model=self.embedding_model,
                inputs=texts,
                parameters={
                    "input_type": "passage",
                    "truncate": "END"
                }
            )
            return [result['values'] for result in results]
        except Exception as e:
            print(f"Error getting embeddings: {e}")
            raise
    
    def add_documents(
        self, 
        documents: List[str], 
        ids: List[str], 
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """Add documents to Pinecone index."""
        if self.index is None:
            raise ValueError("Index not initialized. Call create_index() first.")
        
        if len(documents) != len(ids):
            raise ValueError("Number of documents must match number of IDs")
        
        if metadatas and len(metadatas) != len(documents):
            raise ValueError("Number of metadatas must match number of documents")
        
        try:
            # Get embeddings for documents
            embeddings = self._get_embeddings(documents)
            
            # Prepare vectors for upsert
            vectors = []
            for i, (doc_id, embedding, document) in enumerate(zip(ids, embeddings, documents)):
                metadata = metadatas[i] if metadatas else {}
                metadata["text"] = document  # Store original text in metadata
                
                vectors.append({
                    "id": doc_id,
                    "values": embedding,
                    "metadata": metadata
                })
            
            # Upsert vectors to Pinecone
            self.index.upsert(vectors=vectors)
            print(f"Successfully added {len(documents)} documents to Pinecone")
            
        except Exception as e:
            print(f"Error adding documents to Pinecone: {e}")
            raise
    
    def query(
        self, 
        query_text: str, 
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Query Pinecone for similar documents."""
        if self.index is None:
            raise ValueError("Index not initialized. Call create_index() first.")
        
        try:
            # Get embedding for query
            query_embedding = self._get_embeddings([query_text])[0]
            
            # Prepare query parameters
            query_params = {
                "vector": query_embedding,
                "top_k": top_k,
                "include_metadata": True
            }
            
            if filters:
                query_params["filter"] = filters
            
            # Query Pinecone
            results = self.index.query(**query_params)
            
            # Format results to match the interface
            formatted_results = []
            for match in results['matches']:
                result = {
                    "id": match['id'],
                    "score": match['score'],
                    "document": match['metadata'].get('text', ''),
                    "metadata": {k: v for k, v in match['metadata'].items() if k != 'text'}
                }
                formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            print(f"Error querying Pinecone: {e}")
            raise
    
    def delete_documents(self, ids: List[str]) -> None:
        """Delete documents by their IDs."""
        if self.index is None:
            raise ValueError("Index not initialized. Call create_index() first.")
        
        try:
            self.index.delete(ids=ids)
            print(f"Successfully deleted {len(ids)} documents from Pinecone")
        except Exception as e:
            print(f"Error deleting documents from Pinecone: {e}")
            raise
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the Pinecone index."""
        try:
            if self.index is None:
                return {
                    "name": self.index_name,
                    "type": "Pinecone",
                    "status": "not_initialized",
                    "exists": self.pc.has_index(self.index_name)
                }
            
            stats = self.index.describe_index_stats()
            return {
                "name": self.index_name,
                "type": "Pinecone",
                "status": "initialized",
                "total_vector_count": stats.get('total_vector_count', 0),
                "dimension": stats.get('dimension', 0),
                "index_fullness": stats.get('index_fullness', 0),
                "embedding_model": self.embedding_model
            }
            
        except Exception as e:
            print(f"Error getting Pinecone index info: {e}")
            return {"error": str(e)}