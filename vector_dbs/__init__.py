"""Vector database implementations package."""

from .vector_store import VectorStore
from .chromadb_store import ChromaDBStore

try:
    from .pinecone_store import PineconeStore
except ImportError:
    # Pinecone may not be installed
    pass

__all__ = ["VectorStore", "ChromaDBStore"]

# Add PineconeStore to __all__ if it was successfully imported
try:
    PineconeStore
    __all__.append("PineconeStore")
except NameError:
    pass