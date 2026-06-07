from dotenv import load_dotenv

load_dotenv()

from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    model = "text-embedding-3-large",
    dimensions = 64
)

texts = [
    "Hello this is Swapnil Menkar",
    "Hello your name is dnyan guru",
    "And you all are very beautiful"
]

vector = embeddings.embed_documents(texts)

print(vector)

