from model import *
from dataloading import *

epochs = 10
batch_size = 64
learning_rate = 0.01

for epoch in range(epochs):
    indices = np.random.permutation(train_images.shape[0])
    train_images_shuffled = train_images[indices]
    train_labels_shuffled = train_labels_one_hot[indices]

    for i in range(0, train_images.shape[0], batch_size):
        X_batch = train_images_shuffled[i:i + batch_size]
        Y_batch = train_labels_shuffled[i:i + batch_size]
        
        Z1, A1, Z2, A2 = forward(X_batch)
        
        loss = cross_entropy_loss(A2, Y_batch)
        
        backward(X_batch, Y_batch, Z1, A1, Z2, A2, learning_rate)
    
    print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}')

def predict(X):
    _, _, _, A2 = forward(X)
    return np.argmax(A2, axis=1)

test_preds = predict(test_images)
test_accuracy = np.mean(test_preds == test_labels)

print(f'Test Accuracy: {test_accuracy * 100:.2f}%')
