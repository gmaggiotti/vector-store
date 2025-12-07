import chromadb
import glob
import os

client = chromadb.PersistentClient(path="./chroma_store")
collection = client.get_or_create_collection("my_documents")

# Use glob to find all text files in the content directory
text_files = glob.glob("./my_documents/*.txt")

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

# Add all documents at once
if documents:
    collection.add(documents=documents, ids=ids, metadatas=metadatas)
    print(f"Successfully loaded {len(documents)} documents:")
    for filename in ids:
        print(f"  - {filename}")
else:
    print("No .txt files found in the content folder.")
