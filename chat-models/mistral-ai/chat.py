from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import init_chat_model


model = init_chat_model("mistral-small-2506")
response = model.invoke("What is generative ai?")

print(response.content)

