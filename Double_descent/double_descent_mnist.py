import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt


tf.random.set_seed(42)                  # setting the random seed for reproducibility. this is done to ensure that the results of the code are consistent across different runs. By setting the random seed, we can control the randomness in the training process, such as weight initialization and data shuffling, which can lead to more stable and comparable results when experimenting with different model architectures or hyperparameters.
np.random.seed(42)                   # setting the random seed for numpy to ensure reproducibility of results when using numpy functions that involve randomness, such as shuffling data or generating random numbers. This helps in achieving consistent results across different runs of the code, making it easier to compare and analyze the performance of different models or configurations.

# dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()  # loading the MNIST dataset, which consists of 60,000 training images and 10,000 test images of handwritten digits (0-9). The dataset is split into training and testing sets, where x_train and y_train contain the training images and their corresponding labels, while x_test and y_test contain the test images and their labels. This dataset is commonly used for image classification tasks in machine learning.

# Flatten and normalize
# each image in mnist was 28x28 pixels, so we reshape it to a 1D array of 784 pixels (28*28) and normalize the pixel values to be between 0 and 1 by dividing by 255.0. This preprocessing step is important for training neural networks, as it helps to improve convergence and performance by ensuring that the input data is in a consistent format and scale.
x_train = x_train.reshape(-1, 28 * 28).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28 * 28).astype("float32") / 255.0


n_samples = 10000              # number of samples to use from the training set. This is done to reduce the computational load and speed up the training process, especially when experimenting with different model architectures or hyperparameters. By using a subset of the training data, we can still achieve good performance while saving time and resources during training.


indices = np.random.permutation(len(x_train))[:n_samples]   # give a random permutation of indices for the training data and selecting the first n_samples indices. 

x_train = x_train[indices]
y_train = y_train[indices]

# adding 15% noise
noise_level = 0.15
n_noise = int(noise_level * n_samples)  # number of samples to be noised

noise_indices = np.random.permutation(n_samples)[:n_noise]

y_train_noisy = y_train.copy()
y_train_noisy[noise_indices] = np.random.randint(0, 10, size=n_noise)  # adding noise to the labels by randomly assigning new labels to a subset of the training data. This is done to simulate real-world scenarios where data may be noisy or mislabeled, and to evaluate the robustness of the model to such noise. By introducing noise into the training labels, we can assess how well the model can learn and generalize from imperfect data.

mean = np.mean(x_train, axis=0)  
std = np.std(x_train, axis=0) + 1e-8  

x_train = (x_train - mean) / std
x_test = (x_test - mean) / std

# one hot encoding -> converting integer labels to binary vector of size 10
y_train_cat = tf.keras.utils.to_categorical(y_train_noisy, 10)    # size will be (10000, 10) where each row corresponds to a one-hot encoded label for the noisy training data. This is done to prepare the labels for training a neural network, as many machine learning algorithms require the labels to be in a one-hot encoded format for multi-class classification tasks. By converting the integer labels to binary vectors, we can effectively train the model to predict the correct class for each input sample.
y_test_cat = tf.keras.utils.to_categorical(y_test, 10)

# model

# ==========================================
# Model
# ==========================================

def build_model(neurons):

    model = Sequential()

    model.add(Dense(neurons, activation='relu', input_shape=(28 * 28,)))  # here input shape is (28*28,) because we have flattened the images into 1D arrays of size 784. we have used (784,) instead of (784) 
    model.add(Dense(10, activation='softmax'))  # output layer with 10 neurons for the 10 classes (digits 0-9) and softmax activation function to convert the output into probabilities.

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])  # compiling the model with the Adam optimizer, categorical cross-entropy loss function (suitable for multi-class classification), and accuracy as the evaluation metric.

    return model

# neurons size
neuron_list = [2, 4, 10, 50, 100, 200, 500, 750, 1000, 5000, 10000]  

train_errors = []
test_errors = []

for neurons in neuron_list:
    print(f"Running model with neurons: {neurons} ")

    model = build_model(neurons)  # building the model with the specified number of neurons in the hidden layer.
    model.fit(x_train, y_train_cat, epochs=1000, batch_size=64, verbose=1)  
    
    # training accuracy and test accuracy
    train_loss, train_acc = model.evaluate(x_train, y_train_cat, verbose=1)  # evaluating the model on the training data to get the training loss and accuracy.

    test_loss, test_acc = model.evaluate(x_test, y_test_cat, verbose=1)  # evaluating the model on the test data to get the test loss and accuracy.

    train_errors.append(1 - train_acc)  # calculating the training error as 1 - training accuracy and appending it to the train_errors list.
    test_errors.append(1 - test_acc)  # calculating the test error as 1 - test accuracy and appending it to the test_errors list.

# plot

plt.figure(figsize=(10, 6))    

plt.plot(neuron_list, train_errors, label='Train Error', color='blue')  
plt.plot(neuron_list, test_errors, label='Test Error', color='orange')

# interpolating threshold
plt.axvline(x=n_samples, color='red', linestyle='--', label='Interpolation Threshold')  # adding a vertical dashed red line at the interpolation threshold (number of samples) to indicate where the model transitions from underfitting to overfitting.

plt.xscale('log')  # setting the x-axis to a logarithmic scale to better visualize the range of neuron sizes.

plt.xlabel('Number of Neurons (log scale)')  
plt.ylabel('Error')

plt.title('Double Descent Phenomenon on MNIST Dataset')  
plt.ylim(0, 1)    # setting y range to 0-1 for better visualization of error rates.

plt.legend()
plt.savefig('double_descent_mnist_plot.png')  

