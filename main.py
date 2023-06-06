import os
from dotenv import load_dotenv
import openai
import pinecone
from datasets import load_dataset
from tqdm.auto import tqdm  # this is our progress bar

load_dotenv()

openai.api_key = os.getenv("OPEN_AI_API_KEY")
MODEL = "text-embedding-ada-002"

pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment=os.getenv("PINECONE_ENV")
)

# Create a Pinecone index
if 'semantic-search' not in pinecone.list_indexes():
    print('Creating pinecone index...')
    pinecone.create_index('semantic-search', dimension=1536)

# Connect to the index
index = pinecone.Index('semantic-search')

quora_dataset = load_dataset('quora', split='train[240000:320000]')
questions = [q['text'] for q in quora_dataset['questions']]
questions = list(set(questions))  # remove duplicates

batch_size = 32  # process everything in batches of 32
for i in tqdm(range(0, len(questions), batch_size)):
    # set end position of batch
    i_end = min(i+batch_size, len(quora_dataset['questions']))
    # get batch of lines and IDs
    lines_batch = quora_dataset['questions'][i: i+batch_size][i]['text']
    print(lines_batch)
    ids_batch = [str(n) for n in range(i, i_end)]
    # create embeddings
    res = openai.Embedding.create(input=lines_batch, engine=MODEL)
    embeds = [record['embedding'] for record in res['data']]
    # prep metadata and upsert batch
    meta = [{'text': line} for line in lines_batch]
    to_upsert = zip(ids_batch, embeds, meta)
    # upsert to Pinecone
    index.upsert(vectors=list(to_upsert))

query = input("Enter a query: ")

xq = openai.Embedding.create(input=query, engine=MODEL)['data'][0]['embedding']
res = index.query([xq], top_k=5, include_metadata=True)

for match in res['matches']:
    print(f"{match['score']:.2f}: {match['metadata']['questions']}")