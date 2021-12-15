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
noise_factor = 0.1

(ds_train, ds_test), info = tfds.load('mnist', split=['train', 'test'], with_info=True, as_supervised=True)
#fig = tfds.show_examples(ds_train, info)

ds_train = ds_train.take(6000)
ds_test = ds_test.take(1000)
# Preprocessing the two datasets
ds_train = preprocessing_data(ds_train, noise_factor)
ds_test = preprocessing_data(ds_test, noise_factor)

tf.keras.backend.clear_session()

num_epochs = 10 # training epochs
learning_rate = 0.001

# Initialize the autoencoder
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
    predictions = None # For later visualisation
    inputs = None # For later visualisation
    epoch_loss_agg = np.empty(0)
    for input, target in ds_train:
        train_loss, predictions, inputs = train_step(encoder, input, target, global_loss_function, optimizer)
        epoch_loss_agg = np.append(epoch_loss_agg, train_loss)

    # Visualisation for every epoch
    n = 5
    plt.figure(figsize=(10, 9))
    for i in range(n):
        # first the original with noise picture
        ax = plt.subplot(3, n, i + 1)
        plt.title("original + noise")
        plt.imshow(tf.squeeze(inputs[i]))
        # second the prediction from the encodert
        ax = plt.subplot(3, n, i + n + 1)
        plt.title("prediction")
        plt.imshow(tf.squeeze(predictions[i]))
        # finally the corresponding target
        ax = plt.subplot(3, n, i + 2 * n + 1)
        plt.title("target")
        plt.imshow(tf.squeeze(target[i]))
    plt.suptitle(f"Noisy images after {epoch} epochs")
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
