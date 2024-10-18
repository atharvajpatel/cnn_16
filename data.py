import torch
import torchvision.datasets as datasets 
import matplotlib.pyplot as plt
import numpy as np

mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=None)

image, label = mnist_trainset[0]

image_np = np.array(image)

print(f'Label: {label}')
plt.imshow(image_np, cmap='gray')
plt.title(f'Label: {label}')
plt.show()
