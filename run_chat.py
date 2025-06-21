from chat import chatbot_response

print("🤖 Customer Support Chatbot (type 'quit' to exit)\n")

while True:
    user_input = input("You: ")

    if user_input.strip().lower() == "quit":
        print("Bot: Goodbye! Have a great day.")
        break

    reply = chatbot_response(user_input)
    print("Bot:", reply)
    
    
    
    
    
    