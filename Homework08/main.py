"""
Main module for Homework 08

Created: 14.12.21, 16:01

Author: LDankert
"""

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
from processing import preprocessing_data, train_step, test
from autoencoder import Autoencoder

# hyper parameter for noise
noise_factor = 0

(ds_train, ds_test), info = tfds.load('mnist', split=['train', 'test'], with_info=True, as_supervised=True)
#fig = tfds.show_examples(ds_train, info)

print(ds_train)
ds_train = preprocessing_data(ds_train, noise_factor)
ds_test = preprocessing_data(ds_test, noise_factor)
print(ds_train)

tf.keras.backend.clear_session()

num_epochs = 10
learning_rate = 0.001

encoder = Autoencoder()
# Initialize the loss function.
global_loss_function = tf.keras.losses.CategoricalCrossentropy()
# Initialize the optimizer:
optimizer = tf.keras.optimizers.Adam(learning_rate)

# Initialize numpy arrays for later visualization
train_losses = np.empty(0)
train_accuracies = np.empty(0)

test_losses = np.empty(0)
test_accuracies = np.empty(0)

# testing once before we begin
test_loss, test_accuracy = test(encoder, ds_test, global_loss_function)
test_losses = np.append(test_losses, test_loss)
test_accuracies = np.append(test_accuracies, test_accuracy)

# check how model performs on train data once before we begin
train_loss, _ = test(encoder, ds_test, global_loss_function)
train_losses = np.append(train_losses, train_loss)

# We train for num_epochs epochs.
for epoch in range(num_epochs):
    # Display accuracy at the beginning of each epoch
    print(f'Epoch: {epoch} starting with test accuracy {test_accuracies[-1]}')

    # Iterate over the batches of the dataset
    prediction = None
    epoch_loss_agg = np.empty(0)
    for input, target in ds_train:
        train_loss, prediction = train_step(encoder, input, target, global_loss_function, optimizer)
        epoch_loss_agg = np.append(epoch_loss_agg, train_loss)

    n = 10
    plt.figure(figsize=(20, 2))
    for i in range(n):
        ax = plt.subplot(1, n, i + 1)
        plt.title("original + noise")
        plt.imshow(tf.squeeze(prediction[i]))
        plt.gray()
    plt.show()


    # Track training loss
    train_losses = np.append(train_losses, tf.reduce_mean(epoch_loss_agg))

    # Computing train accuracy
    _, train_accuracy = test(encoder, ds_train, global_loss_function)
    train_accuracies = np.append(train_accuracies, train_accuracy)

    # Display train accuracy
    print(f'Epoch: {epoch} finishing with train accuracy {train_accuracies[-1]}')
    print(" ")

    # Computing test loss and accuracy
    test_loss, test_accuracy = test(encoder, ds_test, global_loss_function)
    test_losses = np.append(test_losses, test_loss)
    test_accuracies = np.append(test_accuracies, test_accuracy)

# Visualize accuracy and loss for training and test data
plt.figure()
line1, = plt.plot(train_losses)
line2, = plt.plot(test_losses)
line3, = plt.plot(test_accuracies)
line4, = plt.plot(train_accuracies)
plt.xlabel("Training steps")
plt.ylabel("Loss/Accuracy")
plt.legend((line1, line2, line3, line4), ("training loss", "test loss", "test accuracy", "train accuracy"))
plt.show()
