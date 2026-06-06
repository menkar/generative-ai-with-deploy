from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-5", temperature=0)

response = model.invoke("What is generative ai")

print(response.content)
