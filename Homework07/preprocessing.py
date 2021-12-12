"""
Preprocessing methods

Created: 11.12.21, 18:54

Author: LDankert
"""
import tensorflow as tf
import numpy as np


# Generator for white noise, needs two integer for sequence length and number of samples
def integration_task(seq_len, num_samples):
    for i in range(num_samples):
        # generating noise error
        noise = np.random.normal(size=seq_len)
        # calculates target
        target = int(np.sum(noise, axis=-1) >= 0)
        # handle empty dimensions
        np.expand_dims(noise, -1)
        np.expand_dims(target, -1)
        yield noise, target


# Function fo preprocessing the dataset
def preprocess_dataset(dataset):
    # shuffle the datasets
    dataset = dataset.shuffle(buffer_size=1000)
    # batch the datasets
    dataset = dataset.batch(64)
    # prefetch the datasets
    dataset = dataset.prefetch(12)
    return dataset
