from warnings import filters

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers
import numpy as np
import matplotlib.pyplot as plt

# reproducibility
np.random.seed(42)
tf.random.set_seed(42)  # Set the random seed for TensorFlow. it means that the random operations in TensorFlow will produce the same results each time you run the code, which is useful for debugging and comparing results.

# dataset cifar10
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()              # size of x_train is (50000, 32, 32, 3) and size of y_train is (50000, 1)

# normalize the data
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0


# data augmentation
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomTranslation(0.1, 0.1)
])

def residual_block(x, filters, stride=1):

    shortcut = x

    x = layers.Conv2D(filters, kernel_size=3, strides=stride, padding='same', use_bias=False)(x)

    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(filters, kernel_size=3, strides=1, padding='same', use_bias=False)(x)

    x = layers.BatchNormalization()(x)

    # matching dimensions
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, kernel_size=1, strides=stride, padding='same',use_bias=False)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([x, shortcut])

    x = layers.ReLU()(x)

    return x

def build_model(use_dropout=True, weight_decay=0, use_augmentation=False):

    inputs = layers.Input(shape=(32,32,3))

    x = inputs

    # Data augmentation
    if use_augmentation:
        x = data_augmentation(x)

    # Initial layer
    x = layers.Conv2D(64, kernel_size=3, padding='same',use_bias=False, kernel_regularizer=regularizers.l2(weight_decay))(x)

    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Residual blocks
    x = residual_block(x, 64)
    x = residual_block(x, 64)

    x = residual_block(x, 128, stride=2)
    x = residual_block(x, 128)

    x = residual_block(x, 256, stride=2)
    x = residual_block(x, 256)

    # Global average pooling
    x = layers.GlobalAveragePooling2D()(x)

    # Dropout
    if use_dropout:
        x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(10,activation='softmax',kernel_regularizer=regularizers.l2(weight_decay))(x)

    model = models.Model(inputs, outputs)

    return model


def run_experiment(use_aug, weight_decay, dropout):

    # Build model
    model = build_model(use_dropout=dropout, weight_decay=weight_decay, use_augmentation=use_aug)

    
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.1,momentum=0.9), loss='sparse_categorical_crossentropy',metrics=['accuracy'])

    train_acc_list = []
    test_acc_list = []
    steps = []

    epochs = 500

    for epoch in range(epochs):

        model.fit(x_train,y_train, epochs=1,batch_size=128,verbose=1)

        # Train accuracy
        train_loss, train_acc = model.evaluate(x_train, y_train, verbose=0)

        # Test accuracy
        test_loss, test_acc = model.evaluate(x_test,y_test, verbose=0)

        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

        steps.append(epoch)

        print("Epoch:", epoch, "Train Accuracy:", train_acc, "Test Accuracy:", test_acc)
    return steps, train_acc_list, test_acc_list



print("Experiment 1")

s1, tr1, val1 = run_experiment(use_aug=True, weight_decay=1e-4,dropout=True)

print("Experiment 2")

s2, tr2, val2 = run_experiment(use_aug=False, weight_decay=1e-4,dropout=False)


print("Experiment 3")

s3, tr3, val3 = run_experiment(use_aug=False, weight_decay=0,dropout=False)

#plotting the results
plt.figure(figsize=(8,6))

plt.plot(s1, val1, 'o-', label='test(w aug, wd, dropout)')
plt.plot(s1, tr1, 'o-', label='train(w aug, wd, dropout)')

plt.plot(s2, val2, 'D-', label='test(w/o aug, wd)')
plt.plot(s2, tr2, 'D-', label='train(w/o aug, wd)')

plt.plot(s3, val3, '-', label='test(w/o aug, wd, dropout)')
plt.plot(s3, tr3, '-', label='train(w/o aug, wd, dropout)')

plt.xlabel("training epochs")
plt.ylabel("accuracy")

plt.legend()

plt.grid()

plt.show()






