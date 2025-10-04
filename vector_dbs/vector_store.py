from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class VectorStore(ABC):
    """Abstract base class for vector store implementations."""
    
    @abstractmethod
    def add_documents(
        self, 
        documents: List[str], 
        ids: List[str], 
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """Add documents to the vector store.
        
        Args:
            documents: List of document texts
            ids: List of unique document IDs
            metadatas: Optional list of metadata dictionaries
        """
        pass
    
    @abstractmethod
    def query(
        self, 
        query_text: str, 
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Query the vector store for similar documents.
        
        Args:
            query_text: The search query
            top_k: Number of results to return
            filters: Optional filters to apply
            
        Returns:
            List of dictionaries containing results with scores, documents, and metadata
        """
        pass
    
    @abstractmethod
    def delete_documents(self, ids: List[str]) -> None:
        """Delete documents by their IDs.
        
        Args:
            ids: List of document IDs to delete
        """
        pass
    
    @abstractmethod
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection/index.
        
        Returns:
            Dictionary containing collection statistics
        """
        pass