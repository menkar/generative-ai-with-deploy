from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import init_chat_model


model = init_chat_model("groq:meta-llama/llama-4-scout-17b-16e-instruct")
response = model.invoke("What is generative ai?")

print(response.content)

