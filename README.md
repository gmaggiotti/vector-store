# Vector Store System

A unified interface for working with different vector database implementations (ChromaDB and Pinecone) through a common abstract base class.

## Features

- **Abstract Base Class**: `VectorStore` provides a unified interface for all implementations
- **ChromaDB Implementation**: `ChromaDBStore` for local/persistent vector storage
- **Pinecone Implementation**: `PineconeStore` for cloud-based vector storage
- **Factory Pattern**: Easy instantiation through `VectorStoreFactory`
- **Configuration-Based**: Switch between implementations using configuration files
- **Unified API**: Same methods work across all implementations

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```python
from vector_store_demo import VectorStoreFactory, VectorStoreManager

# Create a ChromaDB store
store = VectorStoreFactory.create_store('chromadb', collection_name='my_docs')

# Create a manager for easier operations
manager = VectorStoreManager(store)

# Add sample data and search
manager.setup_with_sample_data()
manager.search("programming languages")
```

### Configuration-Based Usage

```python
from configurable_vector_store import ConfigurableVectorStore

# Initialize with configuration
config = {
    "store_type": "chromadb",
    "chromadb": {
        "collection_name": "my_documents",
        "persist_directory": "./my_chroma_store"
    }
}

vector_store = ConfigurableVectorStore(config_dict=config)

# Add documents
documents = ["Document 1 text", "Document 2 text"]
ids = ["doc1", "doc2"]
vector_store.add_documents(documents, ids)

# Search
results = vector_store.query("search query", top_k=5)
```

## API Reference

### VectorStore (Abstract Base Class)

```python
class VectorStore(ABC):
    def add_documents(self, documents: List[str], ids: List[str], metadatas: Optional[List[Dict[str, Any]]] = None) -> None
    def query(self, query_text: str, top_k: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]
    def delete_documents(self, ids: List[str]) -> None
    def get_collection_info(self) -> Dict[str, Any]
```

### ChromaDBStore

```python
# Initialize
store = ChromaDBStore(
    collection_name="documents",
    persist_directory="./chroma_store"
)

# Load documents from directory (utility method)
store.load_documents_from_directory("./content", "*.txt")
```

### PineconeStore

```python
# Initialize
store = PineconeStore(
    api_key="your-api-key",  # or None to load from file
    index_name="my-index",
    key_file="./pinecone_key.json",
    embedding_model="multilingual-e5-large"
)

# Create index if needed
store.create_index(cloud="aws", region="us-east-1")
```

## Configuration File Format

Create a `vector_store_config.json` file:

```json
{
  "store_type": "chromadb",
  "chromadb": {
    "collection_name": "my_documents",
    "persist_directory": "./chroma_store"
  },
  "pinecone": {
    "index_name": "my-index",
    "key_file": "./pinecone_key.json",
    "embedding_model": "multilingual-e5-large"
  }
}
```

## File Structure

```
├── vector_store.py              # Abstract base class
├── chromadb_store.py           # ChromaDB implementation
├── pinecone_store.py           # Pinecone implementation
├── vector_store_demo.py        # Factory and demo usage
├── configurable_vector_store.py # Configuration-based usage
├── requirements.txt            # Dependencies
└── vector_store_config.json    # Sample configuration
```

## Examples

### Example 1: Basic ChromaDB Usage

```python
from chromadb_store import ChromaDBStore

# Create store
store = ChromaDBStore(collection_name="test_docs")

# Add documents
documents = ["Hello world", "How are you?"]
ids = ["doc1", "doc2"]
metadatas = [{"type": "greeting"}, {"type": "question"}]

store.add_documents(documents, ids, metadatas)

# Query
results = store.query("hello", top_k=1)
print(results[0]["document"])  # "Hello world"
```

### Example 2: Basic Pinecone Usage

```python
from pinecone_store import PineconeStore

# Create store (requires valid API key)
store = PineconeStore(index_name="test-index")

# Create index
store.create_index()

# Add documents
documents = ["Machine learning basics", "Deep learning concepts"]
ids = ["ml1", "dl1"]

store.add_documents(documents, ids)

# Query
results = store.query("artificial intelligence", top_k=2)
```

### Example 3: Switching Between Stores

```python
from configurable_vector_store import ConfigurableVectorStore

# Start with ChromaDB
vector_store = ConfigurableVectorStore(config_dict={
    "store_type": "chromadb",
    "chromadb": {"collection_name": "test"}
})

# Add some documents
vector_store.add_documents(["test doc"], ["id1"])

# Switch to Pinecone (if configured)
vector_store.switch_store("pinecone")
```

## Running the Examples

```bash
# Run the basic demo
python vector_store_demo.py

# Run the configurable demo
python configurable_vector_store.py
```

## Notes

- **ChromaDB**: Works locally, great for development and testing
- **Pinecone**: Requires API key and internet connection, better for production
- **Embeddings**: Pinecone handles embeddings automatically, ChromaDB uses built-in models
- **Persistence**: ChromaDB persists to disk, Pinecone is cloud-based
- **Filters**: Both support metadata filtering with slightly different syntax

## Error Handling

All implementations include proper error handling and informative error messages. Common issues:

- Missing API keys for Pinecone
- Index not created for Pinecone
- File permission issues for ChromaDB persistence
- Mismatched document/ID/metadata array lengths