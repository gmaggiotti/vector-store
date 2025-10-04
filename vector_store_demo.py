from typing import Union
from vector_dbs.vector_store import VectorStore
from vector_dbs.chromadb_store import ChromaDBStore
from vector_dbs.pinecone_store import PineconeStore


class VectorStoreFactory:
    """Factory class to create vector store instances."""
    
    @staticmethod
    def create_store(store_type: str, **kwargs) -> VectorStore:
        """Create a vector store instance.
        
        Args:
            store_type: Type of store ('chromadb' or 'pinecone')
            **kwargs: Configuration parameters specific to each store type
            
        Returns:
            VectorStore instance
            
        Raises:
            ValueError: If store_type is not supported
        """
        if store_type.lower() == 'chromadb':
            return ChromaDBStore(
                collection_name=kwargs.get('collection_name', 'documents'),
                persist_directory=kwargs.get('persist_directory', './chroma_store')
            )
        elif store_type.lower() == 'pinecone':
            return PineconeStore(
                api_key=kwargs.get('api_key'),
                index_name=kwargs.get('index_name', 'my-index'),
                key_file=kwargs.get('key_file', './pinecone_key.json'),
                embedding_model=kwargs.get('embedding_model', 'multilingual-e5-large')
            )
        else:
            raise ValueError(f"Unsupported store type: {store_type}. Supported types: 'chromadb', 'pinecone'")


class VectorStoreManager:
    """Manager class to handle vector store operations with a unified interface."""
    
    def __init__(self, store: VectorStore):
        """Initialize with a vector store instance.
        
        Args:
            store: VectorStore implementation instance
        """
        self.store = store
    
    
    def search(self, query: str, top_k: int = 3):
        """Search for similar documents."""
        results = self.store.query(query, top_k=top_k)
        
        print(f"\nSearch results for: '{query}'")
        print("-" * 50)
        
        for i, result in enumerate(results, 1):
            print(f"{i}. Score: {result['score']:.4f}")
            print(f"   ID: {result['id']}")
            print(f"   Document: {result['document'][:1000]}...")
            print(f"   Metadata: {result['metadata']}")
            print()
    
    def get_info(self):
        """Display information about the vector store."""
        info = self.store.get_collection_info()
        print("\nVector Store Information:")
        print("-" * 30)
        for key, value in info.items():
            print(f"{key}: {value}")
        print()


def main():
    """Example usage of the vector store system."""
    print("Vector Store System Demo")
    print("=" * 50)
    
    # Example 1: Using ChromaDB
    print("\n1. ChromaDB Example:")
    print("-" * 20)
    
    chroma_store = VectorStoreFactory.create_store(
        'chromadb',
        collection_name='my_documents',
        persist_directory='./chroma_store'
    )
    
    chroma_manager = VectorStoreManager(chroma_store)
    chroma_manager.get_info()
    chroma_manager.search(" check your flight status ")
    
    # Example 2: Using Pinecone (commented out as it requires API key and index setup)
    """
    print("\n2. Pinecone Example:")
    print("-" * 20)
    
    pinecone_store = VectorStoreFactory.create_store(
        'pinecone',
        index_name='demo-index',
        key_file='./pinecone_key.json'
    )
    
    # Create index if it doesn't exist
    pinecone_store.create_index()
    
    pinecone_manager = VectorStoreManager(pinecone_store)
    pinecone_manager.setup_with_sample_data()
    pinecone_manager.get_info()
    pinecone_manager.search("programming languages")
    """


if __name__ == "__main__":
    main()