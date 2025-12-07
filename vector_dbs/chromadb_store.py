import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional
from .vector_store import VectorStore


class ChromaDBStore(VectorStore):
    """ChromaDB implementation of the VectorStore interface."""

    def __init__(
        self,
        collection_name: str = "documents",
        persist_directory: str = "./chroma_store",
    ):
        """Initialize ChromaDB store.

        Args:
            collection_name: Name of the collection
            persist_directory: Directory to persist the database
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(collection_name)

    def add_documents(
        self,
        documents: List[str],
        ids: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Add documents to ChromaDB collection."""
        if len(documents) != len(ids):
            raise ValueError("Number of documents must match number of IDs")

        if metadatas and len(metadatas) != len(documents):
            raise ValueError("Number of metadatas must match number of documents")

        try:
            self.collection.add(documents=documents, ids=ids, metadatas=metadatas)
            print(f"Successfully added {len(documents)} documents to ChromaDB")
        except Exception as e:
            print(f"Error adding documents to ChromaDB: {e}")
            raise

    def query(
        self, query_text: str, top_k: int = 5, filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Query ChromaDB for similar documents."""
        try:
            query_params = {"query_texts": [query_text], "n_results": top_k}

            if filters:
                if "where" in filters:
                    query_params["where"] = filters["where"]
                if "where_document" in filters:
                    query_params["where_document"] = filters["where_document"]

            results = self.collection.query(**query_params)

            # Format results to match the interface
            formatted_results = []
            for i in range(len(results["documents"][0])):
                result = {
                    "id": results["ids"][0][i],
                    "document": results["documents"][0][i],
                    "score": 1
                    - results["distances"][0][
                        i
                    ],  # Convert distance to similarity score
                    "metadata": (
                        results["metadatas"][0][i] if results["metadatas"][0] else {}
                    ),
                }
                formatted_results.append(result)

            return formatted_results

        except Exception as e:
            print(f"Error querying ChromaDB: {e}")
            raise

    def delete_documents(self, ids: List[str]) -> None:
        """Delete documents by their IDs."""
        try:
            self.collection.delete(ids=ids)
            print(f"Successfully deleted {len(ids)} documents from ChromaDB")
        except Exception as e:
            print(f"Error deleting documents from ChromaDB: {e}")
            raise

    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the ChromaDB collection."""
        try:
            count = self.collection.count()
            return {
                "name": self.collection_name,
                "type": "ChromaDB",
                "document_count": count,
                "persist_directory": self.persist_directory,
            }
        except Exception as e:
            print(f"Error getting ChromaDB collection info: {e}")
            return {"error": str(e)}

    def load_documents_from_directory(
        self, directory_path: str, file_pattern: str = "*.txt"
    ) -> None:
        """Load documents from a directory (utility method specific to ChromaDB)."""
        import glob
        import os

        text_files = glob.glob(os.path.join(directory_path, file_pattern))

        documents = []
        ids = []
        metadatas = []

        for file_path in text_files:
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()
                filename = os.path.basename(file_path)

                documents.append(content)
                ids.append(filename)
                metadatas.append(
                    {"filename": filename, "source": file_path, "type": "text_file"}
                )

        if documents:
            self.add_documents(documents, ids, metadatas)
            print(f"Loaded {len(documents)} documents from {directory_path}")
        else:
            print(f"No files matching {file_pattern} found in {directory_path}")
