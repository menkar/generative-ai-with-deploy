from dotenv import load_dotenv

load_dotenv()

from langchain_mistralai import ChatMistralAI

model = ChatMistralAI(model = "mistral-small-2506", temperature=0.9)
messages = []

print("------------------------------------ Welcome type 0 to exit the application-----------------------------------------")
    
while True:
    prompt = input("You: ")
    messages.append(prompt)

    if prompt == "0":
        break

    messages.append(prompt)
    response = model.invoke(prompt)    
    messages.append(response.content)
    print("Bot: ", response.content)

print("Messages : ", messages)