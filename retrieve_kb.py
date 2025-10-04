import chromadb
from chromadb.config import Settings

client = chromadb.PersistentClient(path="./chroma_store")
collection = client.get_or_create_collection("my_documents")


# Query/search 2 most similar results. You can also .get by id
results = collection.query(
    query_texts=["what does gabriel do"],
    n_results=2,
    # where={"metadata_field": "is_equal_to_this"}, # optional filter
    # where_document={"$contains":"search_string"}  # optional filter
)

for article in results["documents"]:
    print(article , "\n")