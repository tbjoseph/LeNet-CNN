## Overview
This project involves implementing the LeNet architecture using PyTorch to perform image recognition tasks on the CIFAR-10 dataset. The goal is to build and train a convolutional neural network to classify images into one of ten classes accurately. This project aims to deepen understanding of neural network architectures and practical applications using PyTorch, focusing on image classification tasks.

## Project Structure
* LeNet Implementation: Implement the foundational LeNet model in the solution.py file. Modify only this file to complete the coding tasks.
* Dataset: Utilize the CIFAR-10 dataset, which consists of 60000 32x32 color images across 10 classes.
* Environment Setup: Ensure Python, PyTorch, and tqdm are installed as per instructions in the provided readme.txt.

## Features
* Data Preprocessing: Rescale image pixels from the 0-255 range to the 0-1 range. The honor section requires normalization using the image's mean and variance.
* Model Architecture: The LeNet model includes sequences of convolutional layers, ReLU activations, max pooling, and fully connected layers.
* Additional Features: Include batch normalization after each convolutional and fully-connected layer (excluding the output layer) and dropout before the output layer to enhance model performance.
* Real-Time Testing and Logging: Use main.py to train the model and evaluate its performance on the CIFAR-10 test set, with logs and analysis included in your report.

## Installation
**PyTorch**: Install PyTorch using either pip or conda based on your environment setup. For Anaconda users:
```
conda install -c pytorch pytorch
```
For other environments:
```
pip install torch
```
Refer to [PyTorch Get Started](https://pytorch.org/get-started/locally/) for detailed installation instructions.


**tqdm**: Install tqdm for progress bars during training and testing. For Anaconda users:
```
conda install tqdm
```
For other environments:
```
pip install tqdm
```

## Dataset Setup
* Download the CIFAR-10 dataset from [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html) and extract it into the code directory.
* Ensure the data folder cifar-10-batches-py is placed in the project directory to load and preprocess data effectively.
