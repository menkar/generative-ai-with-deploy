from dotenv import load_dotenv

load_dotenv()

from langchain_mistralai import ChatMistralAI

model = ChatMistralAI(model="mistral-small-2506", temperature=0.7, max_tokens=20)

response = model.invoke("What is machine learning")

print(response.content)
