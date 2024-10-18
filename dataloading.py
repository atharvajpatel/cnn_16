import numpy as np
import gzip
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder


def load_mnist_images(filename):
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
        return data.reshape(-1, 28*28).astype(np.float32) / 255.0  

def load_mnist_labels(filename):
    with gzip.open(filename, 'rb') as f:
        return np.frombuffer(f.read(), np.uint8, offset=8)

train_images = load_mnist_images('data/MNIST/raw/train-images-idx3-ubyte.gz')
train_labels = load_mnist_labels('data/MNIST/raw/train-labels-idx1-ubyte.gz')
test_images = load_mnist_images('data/MNIST/raw/t10k-images-idx3-ubyte.gz')
test_labels = load_mnist_labels('data/MNIST/raw/t10k-labels-idx1-ubyte.gz')

one_hot_encoder = OneHotEncoder(sparse_output=False, categories='auto')
train_labels_one_hot = one_hot_encoder.fit_transform(train_labels.reshape(-1, 1))

print(f"Training set shape: {train_images.shape}")
print(f"Training labels shape: {train_labels_one_hot.shape}")
