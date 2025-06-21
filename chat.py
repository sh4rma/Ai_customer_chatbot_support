import json
import numpy as np
import pickle

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model and supporting files
model = load_model("chatbot_model.h5")
tokenizer = pickle.load(open("tokenizer.pickle", "rb"))
lbl_encoder = pickle.load(open("label_encoder.pickle", "rb"))

with open("intents.json") as file:
    data = json.load(file)

def chatbot_response(user_input):
    result = tokenizer.texts_to_sequences([user_input])
    result = pad_sequences(result, truncating='post', maxlen=20)

    prediction = model.predict(result)[0]
    max_prob = np.max(prediction)
    tag_index = np.argmax(prediction)
    tag = lbl_encoder.inverse_transform([tag_index])[0]

    if max_prob > 0.7:  # Confidence threshold updated to 0.7
        for intent in data["intents"]:
            if intent["tag"] == tag:
                return np.random.choice(intent["responses"])
    else:
        return "I'm not sure I understand. Can you rephrase that?"
