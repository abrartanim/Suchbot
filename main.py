import nltk
from nltk.stem.lancaster import LancasterStemmer

# nltk.download('punkt')

stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow
import random
import json
import pickle

import pandas as pd

with open("intents.json") as file:
    data = json.load(file)
# print(data["intents"])

try:
    with open("data.pickle", "rb") as f:
       words, labels, training, output = pickle.load(f)


except:
    words = []
    docs_x = []
    docs_y = []
    labels = []

    #isolating the data
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])
        if intent["tag"] not in labels:
            labels.append(intent["tag"])



    words = [stemmer.stem(w.lower()) for w in words if w not in "?"]
    words = sorted(list(set(words)))
    lables = sorted(labels)


    training = []
    ouput = []

    output_empty = [0 for _ in range(len(lables))]


    #bag of words
    for x, doc in enumerate(docs_x):
        bag = []
        wrds = [stemmer.stem(w) for w in doc]
        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)
        ouput_row = output_empty[:]
        ouput_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        ouput.append(ouput_row)

    training = numpy.array(training)
    output = numpy.array(ouput)
    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

tensorflow.compat.v1.reset_default_graph()

#input data
net = tflearn.input_data(shape = [None, len(training[0])])
#neural network (2), fully connected
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)

net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)
try:
    model.load("Model.tflearn")
except:
    model.fit(training, output, n_epoch=1000, batch_size = 8, show_metric = True)
    model.save("Model.tflearn")

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    return numpy.array(bag)

def chat():
    print("Start talking with the bot! ")
    while(True):
        inp = input("YOU: ")
        if inp.lower() == "quit":
            break
        result = model.predict([bag_of_words(inp, words)])
        result_index = numpy.argmax(result)
        tag = labels[result_index]

        # if tag == "query":
        #     pd.read_excel("query.xlsx", index_col= 0)
        #     continue

        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']
        print(random.choice(responses))

chat()












