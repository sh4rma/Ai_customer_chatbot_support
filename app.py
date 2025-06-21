from flask import Flask, request, jsonify, send_from_directory
from chat import chatbot_response
import os

app = Flask(__name__, static_folder='')

# Made by Hritik Sharma

@app.route('/')
def index():
    return send_from_directory('', 'index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get('message', '')
    bot_reply = chatbot_response(user_message)
    return jsonify({'response': bot_reply})

if __name__ == '__main__':
    app.run(debug=True)
