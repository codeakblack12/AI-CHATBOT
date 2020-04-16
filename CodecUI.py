import random
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import pickle
from nltk.stem.lancaster import LancasterStemmer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import json
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout

stemmer = LancasterStemmer
labels = []
words = []
training_data = []
def data_creation():
    with open('intent.json', encoding='cp437') as file:
        data = json.load(file)
        words = pickle.load(open("words.pickle","rb"))
        labels = pickle.load(open("labels.pickle","rb"))
        for intent in data["intents"]:
            for pattern in intent["patterns"]:
                pattern = pattern.replace(("'"),"")
                pattern = pattern.lower()
                wrds = nltk.word_tokenize(pattern)
                for wds in wrds:
                    if wds not in words:
                        words.append(wds)
                        
                if intent["tag"] not in labels:
                    labels.append(intent["tag"])

        pickle_out = open("words.pickle","wb")
        pickle.dump(words, pickle_out)
        pickle_out.close()

        pickle_out = open("labels.pickle","wb")
        pickle.dump(labels, pickle_out)
        pickle_out.close()

def trainer():
    words = pickle.load(open("words.pickle","rb"))
    labels = pickle.load(open("labels.pickle","rb"))

    with open('intent.json', encoding='cp437') as file:
        data = json.load(file)
        for intent in data["intents"]:
            for tag in intent["tag"]:
                pattern = intent["patterns"]
                tag_num = labels.index(tag)
                for pattn in pattern:
                    bag = [0 for _ in range(len(words))]
                    pattn = pattn.replace(("'"),"")
                    pattn = pattn.lower()
                    wrds = nltk.word_tokenize(pattn)
                    for wds in wrds:
                        for i, wd in enumerate(words):
                            if wd == wds:
                                bag[i] = 1
                    training_data.append([bag, tag_num])
                            
                
                       
data_creation()                
trainer()

random.shuffle(training_data)

X = []
Y = []

for features, label in training_data:
    X.append(features)
    Y.append(label)

X = np.array(X)
Y = np.array(Y)

model = Sequential()
model.add(Dense(300))
model.add(Dense(150))
model.add(Dense(25))
model.add(Activation('softmax'))

model.compile(loss = "sparse_categorical_crossentropy", optimizer = "adam", metrics = ['accuracy'])

model.fit(X, Y, batch_size = 5, validation_split=0.2, epochs = 3000)

val_loss, val_acc = model.evaluate(X, Y)
print(val_loss)
print(val_acc)
model.save('CBmodel')

input("Press any key...")
