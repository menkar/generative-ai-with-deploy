from dotenv import load_dotenv
import os

load_dotenv()

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1",
    huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_ACCESS_TOKEN"],
    temperature=0.7
)

model = ChatHuggingFace(llm=llm)

response = model.invoke("What is generative ai?")

print(response.content)