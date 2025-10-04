from pinecone import Pinecone, ServerlessSpec
import json

def load_key(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    return data['pinecone_api_key']

key = load_key('./pinecone_key.json')
print(f'key: {key}')

pc = Pinecone(api_key=key)
my_index = "my-index"

if not pc.has_index(my_index):
    pc.create_index_for_model(
        name=my_index,
        cloud="aws",
        region="us-east-1",
        embed = {
            "model":"llama-text-embed-v2",
            "field_map": {"text":"chunk_text"}
        }
    )
