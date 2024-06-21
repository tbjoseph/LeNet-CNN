import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pickle, tqdm, os
import time


def load_data(data_dir):
    '''
    To load the Cifar-10 Dataset from files and reshape the 
    images arrays from shape [3072,] to shape [3, 32, 32].

    Please follow the instruction on how to load the data and 
    labels at https://www.cs.toronto.edu/~kriz/cifar.html

    Args:
        data_dir: String. The directory where data batches are 
            stored.

    Returns:
        x_train: An numpy array of shape [50000, 3, 32, 32].
            (dtype=np.uint8)
        y_train: An numpy array of shape [50000,].
            (dtype=np.int64)
        x_test: An numpy array of shape [10000, 3, 32, 32].
            (dtype=np.uint8)
        y_test: An numpy array of shape [10000,].
            (dtype=np.int64)
    '''

    ### YOUR CODE HERE

    def unpickle(file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    x_train = []
    y_train = []
    for i in range(1, 6):
        batch = unpickle(os.path.join(data_dir, 'data_batch_{}'.format(i)))
        x_train.append(batch[b'data'])
        y_train += batch[b'labels']

    test_batch = unpickle(os.path.join(data_dir, 'test_batch'))
    x_test = test_batch[b'data']
    y_test = test_batch[b'labels']

    x_train = np.concatenate(x_train).reshape(-1, 3, 32, 32).astype(np.uint8)
    y_train = np.array(y_train).astype(np.int64)
    x_test = x_test.reshape(-1, 3, 32, 32).astype(np.uint8)
    y_test = np.array(y_test).astype(np.int64)

    return x_train, y_train, x_test, y_test


def preprocess(train_images, test_images, normalize=False):
    '''
    To preprocess the data by 
        (1).Rescaling the pixels from integers in [0,255) to 
            floats in [0,1), or 
        (2).Normalizing each image using its mean and variance. 

    If you are working on the honor section, please implement 
        (1) and then (2). 
    If not, please implement (1) only.

    Args:
        train_images: An numpy array of shape [50000, 3, 32, 32].
            (dtype=np.uint8)
        test_images: An numpy array of shape [10000, 3, 32, 32].
            (dtype=np.uint8)
        normalize: Boolean. To control to rescale or normalize 
            the images. (Only for the honor section)

    Returns:
        train_images: An numpy array of shape [50000, 3, 32, 32].
            (dtype=np.float64)
        test_images: An numpy array of shape [10000, 3, 32, 32].
            (dtype=np.float64)
    '''
    ### YOUR CODE HERE

    if not normalize:
        train_images = train_images.astype(np.float64)
        test_images = test_images.astype(np.float64)
        train_images /= 255.0
        test_images /= 255.0

    if normalize:
        train_avg = np.mean(train_images)
        train_std = np.std(train_images)
        train_images = (train_images - train_avg) / train_std
        test_images = (test_images - train_avg) / train_std

    ### END CODE HERE
    return train_images, test_images


class LeNet(nn.Module):
    '''
    To build the LeCun network:
        Inputs --> 
        Convolution (6) --> ReLU --> Max Pooling --> 
        Convolution (16) --> ReLU --> Max Pooling --> 
        Reshape to vector --> 
        Fully-connected (120) --> ReLU -->
        Fully-connected (84) --> ReLU --> Outputs (n_classes).

    You are free to use the listed APIs from torch.nn:
        torch.nn.Conv2d
        torch.nn.MaxPool2d
        torch.nn.Linear
        torch.nn.ReLU (or other activations)

    For the honor section, you may also need:
        torch.nn.BatchNorm2d
        torch.nn.BatchNorm1d
        torch.nn.Dropout

    Refer to https://pytorch.org/docs/stable/nn.html
    for the instructions for those APIs
    '''
    def __init__(self, n_classes=None):
        super(LeNet, self).__init__()
        '''
        Define each layers of the model in __init__() function
        '''

        ### YOUR CODE HERE

        super(LeNet, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=0)
        self.batchnorm1 = nn.BatchNorm2d(6)

        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.batchnorm2 = nn.BatchNorm2d(16)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.fc1 = nn.Linear(16*5*5, 120)
        self.batchnorm3 = nn.BatchNorm1d(120)
        
        self.fc2 = nn.Linear(120, 84)
        self.batchnorm4 = nn.BatchNorm1d(84)
        self.fc3 = nn.Linear(84, n_classes)
        self.dropout = nn.Dropout(0.5)

        ### END CODE HERE
    
    def forward(self, x):
        '''
        Run forward pass of the model defined in the above __init__() function
        Args:
            x: Tensor of shape [None, 3, 32, 32]
            for input images.

        Returns:
            logits: Tensor of shape [None, n_classes].
        '''

        ### YOUR CODE HERE

        # Convolution (6) --> Batch Norm --> ReLU --> Max Pooling --> 
        x = self.pool(F.relu(self.batchnorm1(self.conv1(x))))

        # Convolution (16) --> Batch Norm --> ReLU --> Max Pooling -->
        x = self.pool(F.relu(self.batchnorm2(self.conv2(x))))

        # Reshape to vector --> 
        x = torch.flatten(x, 1)

        # Fully-connected (120) --> Batch Norm --> ReLU -->
        x = F.relu(self.batchnorm3(self.fc1(x)))
        
        # Fully-connected (84) --> Batch Norm --> ReLU --> Dropout --> Outputs (n_classes).
        x = F.relu(self.batchnorm4(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)

        ### END CODE HERE
        return x


class LeNet_Cifar10(nn.Module):
    def __init__(self, n_classes):

        super(LeNet_Cifar10, self).__init__()

        self.n_classes = n_classes
        self.model = LeNet(n_classes=n_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def train(self, x_train, y_train, x_valid, y_valid, batch_size, max_epoch):

        num_samples = x_train.shape[0]
        num_batches = int(num_samples / batch_size)

        num_valid_samples = x_valid.shape[0]
        num_valid_batches = (num_valid_samples - 1) // batch_size + 1

        x_train = torch.from_numpy(x_train).float()
        y_train = torch.from_numpy(y_train)
        x_valid = torch.from_numpy(x_valid).float()
        y_valid = torch.from_numpy(y_valid)

        print('---Run...')
        for epoch in range(1, max_epoch + 1):
            self.model.train()
            # To shuffle the data at the beginning of each epoch.
            shuffle_index = np.random.permutation(num_samples)
            curr_x_train = x_train[shuffle_index]
            curr_y_train = y_train[shuffle_index]

            # To start training at current epoch.
            loss_value = []
            qbar = tqdm.tqdm(range(num_batches))
            for i in qbar:
                batch_start_time = time.time()

                start = batch_size * i
                end = batch_size * (i + 1)
                x_batch = curr_x_train[start:end]
                y_batch = curr_y_train[start:end]

                self.optimizer.zero_grad()
                outputs = self.model(x_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()
                if not i % 10:
                    qbar.set_description(
                        'Epoch {:d} Loss {:.6f}'.format(
                            epoch, loss))

            # To start validation at the end of each epoch.
            self.model.eval()
            correct = 0
            total = 0
            print('Doing validation...', end=' ')
            with torch.no_grad():
                for i in range(num_valid_batches):

                    start = batch_size * i
                    end = min(batch_size * (i + 1), x_valid.shape[0])
                    x_valid_batch = x_valid[start:end]
                    y_valid_batch = y_valid[start:end]

                    outputs = self.model(x_valid_batch)
                    _, predicted = torch.max(outputs.data, 1)
                    total += y_valid_batch.shape[0]
                    correct += (predicted == y_valid_batch).sum().item()

            acc = correct / total
            print('Validation Acc {:.4f}'.format(acc))

    def test(self, X_test, y_test):
        self.model.eval()

        X_test = torch.from_numpy(X_test).float()
        y_test = torch.from_numpy(y_test)

        accs = 0
        for X, y in zip(X_test, y_test):

            outputs = self.model(X.unsqueeze(0))
            _, predicted = torch.max(outputs.data, 1)
            accs += (predicted == y).sum().item()

        accuracy = float(accs) / len(y_test)
        
        return accuracy