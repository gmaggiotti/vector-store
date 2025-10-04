from pinecone import Pinecone, ServerlessSpec
import json

def load_key(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    return data['pinecone_api_key']

key = load_key('./pinecone_key.json')
print(f'key: {key}')

pc = Pinecone(api_key=key)

raw_data = "Movies about seduced by a mysterious woman"

results = pc.inference.embed(
    model="multilingual-e5-large",
    inputs=[raw_data],
    parameters={
        "input_type": "query",
        "truncate": "END"
    }
)

index = pc.Index("movies-walkthrough")

query_vector = results[0]['values']

# 4. Query Pinecone
results = index.query(
    vector=query_vector,
    top_k=5,
    include_metadata=True  # optional: include metadata stored with vectors
)

# 5. Interpret the results
for match in results['matches']:
    print(f"Score: {match['score']}")
    print(f"ID: {match['id']}")
    print(f"Metadata: {match.get('metadata')}")
    print("---------")