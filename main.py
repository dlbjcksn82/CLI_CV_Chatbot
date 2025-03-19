import sys
from chatbot import chat_with_gemma

print("Starting up chat")
print("Type the word quit to end the program")
# Loop to enable the user to continually chat
runLLM = True
while runLLM:
    try:
        query = input("Enter your next message: ").strip()
        if query == "":
            continue
        if query == "quit":
            runLLM = False
        else:
            chat_with_gemma(query)
    except KeyboardInterrupt:
        print("Chatbot Terminated")
        sys.exit(0)
