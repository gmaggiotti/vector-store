import chromadb

# setup Chroma in-memory, for easy prototyping. Can add persistence easily!
client = chromadb.Client()

# Create collection. get_collection, get_or_create_collection, delete_collection also available!
collection = client.create_collection("all-my-documents")

# Add docs to the collection. Can also update and delete. Row-based API coming soon!
collection.add(
    documents=[
        "This is document1",
        "This is document2",
    ],  # we handle tokenization, embedding, and indexing automatically. You can skip that and add your own embeddings as well
    metadatas=[{"source": "notion"}, {"source": "google-docs"}],  # filter on these!
    ids=["doc1", "doc2"],  # unique for each doc
)

# Query/search 2 most similar results. You can also .get by id
results = collection.query(
    query_texts=["This is a query document"],
    n_results=2,
    # where={"metadata_field": "is_equal_to_this"}, # optional filter
    # where_document={"$contains":"search_string"}  # optional filter
)

for result in results["documents"]:
    print(result)

{
    "ids": [["doc1", "doc2"]],
    "embeddings": None,
    "documents": [["This is document1", "This is document2"]],
    "uris": None,
    "included": ["metadatas", "documents", "distances"],
    "data": None,
    "metadatas": [[{"source": "notion"}, {"source": "google-docs"}]],
    "distances": [[0.9026353359222412, 1.035815954208374]],
}
