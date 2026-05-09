import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize pixels to [0,1]
x_train = x_train / 255.0
x_test = x_test / 255.0

activations = ["relu", "tanh", "sigmoid", "gelu"]

depths = [2, 4, 6]

all_test_acc = {}

all_gaps = {}

for depth in depths:
    plt.figure(figsize=(10, 6))
    test_accs = []
    gaps = []

    for activation in activations:

        model = Sequential()
        model.add(Flatten(input_shape=(28, 28)))

        for i in range(depth):
            if activation == "gelu":
                model.add(Dense(128,activation=tf.nn.gelu))

            else:
                model.add(Dense(128,activation=activation))

        # Output layer
        model.add(Dense(10, activation="softmax"))

        model.compile(optimizer=Adam(learning_rate=0.001), loss="sparse_categorical_crossentropy",metrics=["accuracy"])

        history = model.fit(x_train, y_train, epochs=200, batch_size=128,validation_split=0.2, verbose=1)
        
        test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=1)

        test_accs.append(test_accuracy)
        
        # Generalization gap
        train_acc = history.history["accuracy"][-1]
        val_acc = history.history["val_accuracy"][-1]

        gap = train_acc - val_acc
        gaps.append(gap)

        # Plot train/validation curves
        plt.plot(history.history["accuracy"], linestyle="--", label=f"{activation}-train")

        plt.plot(history.history["val_accuracy"], label=f"{activation}-val")

    all_test_acc[depth] = test_accs

    all_gaps[depth] = gaps

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy | Depth={depth}")
    plt.legend()
    plt.grid(True)
    plt.show()

for depth in depths:
    plt.figure(figsize=(6, 4))
    plt.bar(activations, all_gaps[depth])
    plt.ylabel("Gap")
    plt.title(f"Generalization Gap | Depth={depth}")
    plt.grid(True)
    plt.show()


for depth in depths:
    plt.figure(figsize=(6, 4))
    plt.bar(activations, all_test_acc[depth])
    plt.ylabel("Accuracy")
    plt.title(f"Test Accuracy | Depth={depth}")
    plt.grid(True)
    plt.show()