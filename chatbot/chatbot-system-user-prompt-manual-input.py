from dotenv import load_dotenv

load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage



model = ChatMistralAI(model = "mistral-small-2506", temperature=0.9)

print("Choose your AI Mode : ")
print("Press 1 for Angry Mode")
print("Press 2 for Funny Mode")
print("Press 3 for Sad Mode")

choice  = int(input("Enter your choice between 1 to 3 : "))
mode = ''
if choice == 1:
    mode = 'You are an angry AI agent. You respons aggreessively and impatiently.'
elif choice == 2:
    mode = 'You are a funny AI agent. You respond with humor and jokes.'
elif choice == 3:
    mode = 'You are a sad agent. You resond in a depressed and emotional tone.'


messages = [
    SystemMessage(content=mode)
]

print("------------------------------------ Welcome type 0 to exit the application -----------------------------------------")
    
while True:
    prompt = input("You: ")
    messages.append(HumanMessage(content = prompt))

    if prompt == "0":
        break

    messages.append(prompt)
    response = model.invoke(prompt)    
    messages.append( AIMessage(content = response.content))
    print("Bot: ", response.content)

print("Messages : ", messages)