from typing import Union
from vector_dbs.vector_store import VectorStore
from vector_dbs.chromadb_store import ChromaDBStore
from vector_dbs.pinecone_store import PineconeStore


class VectorStoreFactory:
    """Factory class to create vector store instances."""
    
    @staticmethod
    def create_store(store_type: str, store_conf: dict) -> VectorStore:
        """Create a vector store instance.
        
        Args:
            store_type: Type of store ('chromadb' or 'pinecone')
            store_conf: Configuration dictionary for the store
            
        Returns:
            VectorStore instance
            
        Raises:
            ValueError: If store_type is not supported
        """

        if store_type.lower() == 'chromadb':
            return ChromaDBStore(
                collection_name=store_conf.get('collection_name'),
                persist_directory=store_conf.get('persist_directory')
            )
        elif store_type.lower() == 'pinecone':
            return PineconeStore(
                api_key=store_conf.get('api_key'),
                index_name=store_conf.get('index_name', 'my-index'),
                key_file=store_conf.get('key_file', './pinecone_key.json'),
                embedding_model=store_conf.get('embedding_model', 'multilingual-e5-large')
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


def load_store_config(storedb_name: str) -> dict:
    """Load vector store configuration from a JSON file.
    
    Args:
        storedb_name: Name of the vector store to load configuration for.
        
    Returns:
        A dictionary containing the configuration for the specified vector store.
        
    Raises:
        FileNotFoundError: If the configuration file is not found.
        KeyError: If the specified storedb_name is not in the configuration file.
        ValueError: If there is an error parsing the configuration file.
    """
    import json
    from pathlib import Path

    config_path = Path("./store_conf.json")
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with config_path.open('r', encoding='utf-8') as file:
            config = json.load(file)
        if storedb_name not in config:
            raise KeyError(f"Configuration for '{storedb_name}' not found in {config_path}")
        return config[storedb_name]
    except json.JSONDecodeError as e:
        raise ValueError(f"Error parsing JSON configuration file {config_path}: {e}")
    except Exception as e:
        raise ValueError(f"Unexpected error loading configuration from {config_path}: {e}")