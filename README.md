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

# Get store information and search
manager.get_info()
manager.search("your search query")
```

### Direct Usage with Vector Database Classes

```python
from vector_dbs.chromadb_store import ChromaDBStore
from vector_dbs.pinecone_store import PineconeStore

# ChromaDB usage
chroma_store = ChromaDBStore(
    collection_name="my_documents",
    persist_directory="./chroma_store"
)

# Add documents
documents = ["Document 1 text", "Document 2 text"]
ids = ["doc1", "doc2"]
chroma_store.add_documents(documents, ids)

# Search
results = chroma_store.query("search query", top_k=5)
```

## API Reference

### VectorStore (Abstract Base Class)

```python
from vector_dbs.vector_store import VectorStore

class VectorStore(ABC):
    def add_documents(self, documents: List[str], ids: List[str], metadatas: Optional[List[Dict[str, Any]]] = None) -> None
    def query(self, query_text: str, top_k: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]
    def delete_documents(self, ids: List[str]) -> None
    def get_collection_info(self) -> Dict[str, Any] 
```

### ChromaDBStore

```python
from vector_dbs.chromadb_store import ChromaDBStore

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
from vector_dbs.pinecone_store import PineconeStore

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

Create a `vector_store_config.json` file for configuration-based usage:

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

## Project Structure

```
├── vector_dbs/                 # Core vector database implementations
│   ├── __init__.py            # Package initialization
│   ├── vector_store.py        # Abstract base class
│   ├── chromadb_store.py      # ChromaDB implementation
│   └── pinecone_store.py      # Pinecone implementation
├── chromadb_examples/         # ChromaDB specific examples
│   ├── kb_in_memory.py        # In-memory knowledge base example
│   ├── load_kb.py             # Load knowledge base example
│   └── retrieve_kb.py         # Retrieve from knowledge base example
├── pinecone_examples/         # Pinecone specific examples
│   ├── create_index.py        # Create Pinecone index example
│   └── query_db.py            # Query Pinecone database example
├── content/                   # Sample documents for testing
│   ├── gabe.txt
│   ├── status1.txt
│   ├── status2.txt
│   └── status3.txt
├── chroma_store/              # ChromaDB persistence directory
├── vector_store_demo.py       # Main demo with factory pattern
├── requirements.txt           # Python dependencies
├── pinecone_key.json          # Pinecone API key configuration
└── README.md                  # This file
```

## Examples

### Example 1: Basic ChromaDB Usage

```python
from vector_dbs.chromadb_store import ChromaDBStore

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
from vector_dbs.pinecone_store import PineconeStore

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

### Example 3: Using the Factory Pattern

```python
from vector_store_demo import VectorStoreFactory, VectorStoreManager

# Create ChromaDB store using factory
store = VectorStoreFactory.create_store(
    'chromadb',
    collection_name="my_docs",
    persist_directory="./my_chroma_store"
)

# Create manager for easier operations
manager = VectorStoreManager(store)

# Get information about the store
manager.get_info()

# Search for documents
manager.search("your search query", top_k=5)
```

## Running the Examples

### Main Demo
```bash
# Run the main demo with factory pattern
python vector_store_demo.py
```

### ChromaDB Examples
```bash
# Run ChromaDB specific examples
python chromadb_examples/kb_in_memory.py
python chromadb_examples/load_kb.py
python chromadb_examples/retrieve_kb.py
```

### Pinecone Examples
```bash
# Run Pinecone specific examples (requires API key setup)
python pinecone_examples/create_index.py
python pinecone_examples/query_db.py
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