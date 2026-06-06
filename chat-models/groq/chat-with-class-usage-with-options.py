from dotenv import load_dotenv

load_dotenv()

from langchain_groq import ChatGroq

model = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0)

response = model.invoke("What is generative ai")

print(response.content)
