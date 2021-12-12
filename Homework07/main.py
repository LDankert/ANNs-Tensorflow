"""
Main file for Homework 07

Created: 11.12.21, 18:08

Author: LDankert
"""

import numpy as np
import tensorflow as tf
from preprocessing import preprocess_dataset, integration_task

sequence_length = 5     # will be imported from preprocessing module
number_of_samples = 200 # will be imported from preprocessing module

# Iterates over integration_task with specific values
def my_integration_task():
    yield next(integration_task(sequence_length, number_of_samples))

# generating datasets output_signature adds metadata to the dataset like dtype
ds = tf.data.Dataset.from_generator(my_integration_task, output_signature=tf.TensorSpec(shape=(5,1),dtype=tf.float32))

# preprocessing the dataset
ds = preprocess_dataset(ds)

#print(next(my_integration_task())[0])

for element in ds:
    print(element)