import json
import numpy as np
import pickle

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

# Load intents
with open("intents.json") as file:
    data = json.load(file)

training_sentences = []
training_labels = []
labels = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        training_sentences.append(pattern)
        training_labels.append(intent["tag"])
    if intent["tag"] not in labels:
        labels.append(intent["tag"])

lbl_encoder = LabelEncoder()
lbl_encoder.fit(training_labels)
training_labels = lbl_encoder.transform(training_labels)

vocab_size = 1000
embedding_dim = 16
max_len = 20
oov_token = "<OOV>"

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
tokenizer.fit_on_texts(training_sentences)
sequences = tokenizer.texts_to_sequences(training_sentences)
padded_sequences = pad_sequences(sequences, truncating='post', maxlen=max_len)

model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, input_length=max_len))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(labels), activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy',
            optimizer=SGD(learning_rate=0.01, momentum=0.9, nesterov=True),
            metrics=['accuracy'])

model.fit(padded_sequences, np.array(training_labels), epochs=200, batch_size=5)

model.save("chatbot_model.h5")
pickle.dump(tokenizer, open("tokenizer.pickle", "wb"))
pickle.dump(lbl_encoder, open("label_encoder.pickle", "wb"))


