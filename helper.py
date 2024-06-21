import os
import pickle
import numpy as np


def train_valid_split(x_train, y_train, split_index=45000):
    """Split the original training data into a new training dataset
    and a validation dataset.

    Args:
        x_train: An array of shape [50000, 32, 32, 3].
        y_train: An array of shape [50000,].
        split_index: An integer.

    Returns:
        x_train_new: An array of shape [split_index, 32, 32, 3].
        y_train_new: An array of shape [split_index,].
        x_valid: An array of shape [50000-split_index, 32, 32, 3].
        y_valid: An array of shape [50000-split_index,].
    """
    x_train_new = x_train[:split_index]
    y_train_new = y_train[:split_index]
    x_valid = x_train[split_index:]
    y_valid = y_train[split_index:]

    return x_train_new, y_train_new, x_valid, y_valid

