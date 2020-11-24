import time
import init_data
from nltk.tokenize import word_tokenize
import json
import random
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import tensorflow as tf
import numpy as np
import nltk
from nltk.stem import LancasterStemmer
import os
import argparse
import chatbot_train
stemmer = LancasterStemmer()

vocab, labels, training, output = init_data.load_data()


def bag_of_words(sentence, vocab):
    bag = [0 for _ in range(len(vocab))]
    s_words = nltk.word_tokenize(sentence)
    s_words = [stemmer.stem(word.lower()) for word in s_words]
    for se in s_words:
        for i, w in enumerate(vocab):
            if w == se:
                bag[i] = 1
    # print(bag)
    return np.array(bag)

# picks the highest probability and returns the index.


def greedy_decoder(results):
    result_index = np.argmax(results)
    return result_index


def random_response(tag):
    for tg in init_data.data["intents"]:
        if tg["tag"] == tag:
            responses = tg["responses"]
    return random.choice(responses)


def chat():
    model = load_model("chatbot_model.h5")
    print("Start talking with the bot! (type quit to stop) ")
    with open("outputconvo.txt", "a") as file:
        while True:
            inp = input("You: ")
            if inp.lower() == "quit":
                break
            file.write('HUMAN ++++ ' + inp + '\n')
            res = bag_of_words(inp, vocab)
            # returns a row with the probability of response being one of our labels
            # looks like -- [0.01 0.2 0.0002 0.3 0.004 .... etc]
            results = model.predict(np.array([res]))[0]
            # we do a greedy decode to find the highset probable label.
            results_index = greedy_decoder(results)
            tag = labels[results_index]
            response = random_response(tag)
            print(response)
            file.write('BOT ++++ ' + response + '\n')
        file.write('=============================================\n')


def train():
    chatbot_train.train()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices={'train', 'chat'},
                        default='train', help="mode. if not specified, it's in the train mode")
    args = parser.parse_args()

    if args.mode == 'train':
        train()
    elif args.mode == 'chat':
        chat()


if __name__ == '__main__':
    main()
