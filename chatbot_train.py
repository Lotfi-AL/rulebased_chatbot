from packaging import version
from datetime import datetime
import math
import init_data
import numpy as np
import tensorflow as tf


print("TensorFlow version: ", tf.__version__)
assert version.parse(tf.__version__).release[0] >= 2, \
    "This notebook requires TensorFlow 2.0 or above."


def train():
    log_dir = "logs\\fit\\" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1)

    vocab, labels, training, output = init_data.load_data()
    len_x = math.floor(len(training)*0.8)
    len_y = math.floor(len(output)*0.8)
    print(len_x)
    print(len_y)
    x_train, y_train = training[:len_x], output[:len_y]
    x_test, y_test = training[len_x:], output[len_y:]

    print(training[0])
    print(output[0])

    # Create model - 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons
    # equal to number of intents to predict output intent with softmax
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(128, input_shape=(
        len(training[0]),), activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(len(output[0]), activation='softmax'))

    sgd = tf.keras.optimizers.SGD(
        lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd, metrics=['accuracy'])
    hist = model.fit(np.array(x_train), np.array(y_train), epochs=1000,
                     batch_size=8, verbose=1,
                     validation_data=(np.array(x_test), np.array(y_test)),
                     callbacks=[tensorboard_callback],
                     )
    print("Average test loss: ", np.average(hist.history['loss']))

    model.save("chatbot_model.h5", hist)
