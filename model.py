import numpy as np
from helperFunc import *

input_size = 28 * 28 
hidden_size = 128
output_size = 10

np.random.seed(40)

W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
b1 = np.zeros((1, hidden_size))

W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
b2 = np.zeros((1, output_size))


def forward(X):
    Z1 = np.dot(X, W1) + b1
    A1 = relu(Z1)

    Z2 = np.dot(A1, W2) + b2
    A2 = softmax(Z2)
    
    return Z1, A1, Z2, A2

def backward(X, Y, Z1, A1, Z2, A2, learning_rate=0.001):
    global W1, b1, W2, b2
    
    dZ2 = cross_entropy_derivative(A2, Y)
    dW2 = np.dot(A1.T, dZ2) / X.shape[0] 
    db2 = np.sum(dZ2, axis=0, keepdims=True) / X.shape[0]
    
    dA1 = np.dot(dZ2, W2.T) 
    dZ1 = dA1 * relu_derivative(Z1)
    dW1 = np.dot(X.T, dZ1) / X.shape[0]
    db1 = np.sum(dZ1, axis=0, keepdims=True) / X.shape[0]
    
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
