# Overview

## Data 
1. Load the data in from torch and torchvision 
2. Split into train test split 
3. One hot encode train labels ---> [5] is now [0,0,0,0,0,1,0,0,0,0]
4. Print dimensions

## Helper functions
1. Relu forward and backwards
2. Write out softmax to go from logits to labels
3. Cross entropy loss both forward and backwards

## Model
1. Define input size (image size), hidden size (128 is standard for MNIST), and output (number of classifications so 10)
2. Define seed for consistency 
3. Initialize weights and biases randomly 
    a. Model is too simple for layer/batchnorm  
    b. Use He initialization {multiply weights by root(2/n)} for faster convergence  
    c. Define them globally to be used everywhere  
4. Forward Z1 --> A1 --> Z2 --> A2 using linear equation and activation functions
5. Backwards from loss to dZ2, dA1, dZ1  
    a. Thus can chain rule to corresponding weights and biases at that layer to find dW1, db1, dW2, db2  
6. Update weights and biases as lr * gradient for each one respectively

## Training
1. Define lr, epochs, and batch_size (all standard values)
2. Shuffle order of train_images and labels for each epochs
3. Iterate through all batches where i starts at 0 and increases by batch_size  
    a. Set the X and Y for that batch  
4. Go forwards with X batch
5. Compute loss with Y batch
6. Go backwards in the batch
7. Print loss per epochs
8. Calculate prediction on test set by putting those images in the trained forward network  
    a. Argmax to turn one hot labels into actual prediction ---> [0,0,0,0,0,1,0,0,0,0] is now [5]
9. Calculate accuracy of test set with given model and print it at the end
