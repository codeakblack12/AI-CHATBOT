import json
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
import nltk
import pickle
import numpy as np
import random

words = pickle.load(open("words.pickle","rb"))
labels = pickle.load(open("labels.pickle","rb"))
new_model = tf.keras.models.load_model('CBmodel')

while True:
    bag = [0 for _ in range(len(words))]
    message = input("Message: ")
    message = message.replace(("'"),"")
    message = message.lower()
    message = nltk.word_tokenize(message)
    for msg in message:
        for i, wd in enumerate(words):
            if wd == msg:
                bag[i] = 1
    bag = np.array([bag])
    predictions = new_model.predict(bag)
    for prediction in predictions:
        reply = labels[np.argmax(prediction)]
    with open('intent.json', encoding='cp437') as file:
        data = json.load(file)
        for intent in data["intents"]:
            for tag in intent["tag"]:
                if tag == reply:
                    response = intent["responses"]
                    print("Reply: " + str(response[random.randrange(len(response))]))
                        
    
        
