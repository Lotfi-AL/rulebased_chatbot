import pickle
import nltk
import numpy as np
import json

from nltk.stem import LancasterStemmer
stemmer = LancasterStemmer()

data = ""

with open("intents.json") as file:
    data = json.load(file)


def load_data():
    try:
        with open("data.pickle", "rb") as f:
            vocab, labels, training, output = pickle.load(f)

    except:
        vocab = []
        labels = []
        # each entry in docs_x corresponds to an entry in docs_y
        # docs_x entry is pattern
        # docs_y entry is the intent so we know how to classify each of our patterns.
        docs_x = []
        docs_y = []

        # first intent
        for intent in data["intents"]:
            # then pattern
            for pattern in intent["patterns"]:
                # tokenize
                words = nltk.word_tokenize(pattern)
                vocab.extend(words)
                docs_x.append(stemmer.stem(pattern))
                docs_y.append(intent["tag"])

            if intent["tag"] not in labels:
                labels.append(intent["tag"])
        # check that lower case is same as uppercase.
        # Stemm all words.
        vocab = [stemmer.stem(w.lower())
                 for w in vocab if (w != "j"and w != ".")]
        # remove duplicates
        vocab = sorted(list(set(vocab)))

        labels = sorted(labels)

        # we create a bag of words that represent any given word in a pattern
        # known as one hot encoded

        training = []
        output = []

        out_empty = [0 for _ in range(len(labels))]

        for x, doc in enumerate(docs_x):
            bag = []
            # go through our vocabulary
            for word in vocab:
                # if a word from vocab is present in our doc
                # we add a 1, otherwise a 0
                if word in doc:
                    bag.append(1)
                else:
                    bag.append(0)

            output_row = list(out_empty)
            output_row[labels.index(docs_y[x])] = 1

            training.append(bag)
            output.append(output_row)

        print("training data created")

        with open("data.pickle", "wb") as f:
            pickle.dump((vocab, labels, training, output), f)

    return vocab, labels, training, output
