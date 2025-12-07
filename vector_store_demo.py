from vector_store_factory import (
    VectorStoreFactory,
    VectorStoreManager,
    load_store_config,
)


def main():
    vectordb_name = "chromadb"  # Change to 'pinecone' to test PineconeStore
    store_conf = load_store_config(vectordb_name)

    """Example usage of the vector store system."""
    print("Vector Store System Demo")
    print("=" * 50)

    # Example 1: Using ChromaDB
    print("\n1. ChromaDB Example:")
    print("-" * 20)

    chroma_store = VectorStoreFactory.create_store(vectordb_name, store_conf)

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
