import numpy as np

# Sigmoid activation function and its derivative


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)

# Mean Squared Error loss function and its derivative


def mse_loss(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))


def mse_loss_derivative(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size


# Initialize parameters
input_size = 1  # Number of input neurons
hidden_size = 5  # Number of hidden neurons
output_size = 1  # Number of output neurons
lr = 0.1  # Learning rate

# Randomly initialize weights and biases
W1 = np.random.randn(input_size, hidden_size)
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size)
b2 = np.zeros((1, output_size))

# Training data
X = np.array([[i] for i in range(1, 6)])  # Inputs: 1, 2, 3, 4, 5
y = np.array([[i] for i in range(2, 7)])  # Expected outputs: 2, 3, 4, 5, 6

# Training loop
for epoch in range(1000):
    # Forward pass
    hidden_input = np.dot(X, W1) + b1
    hidden_output = sigmoid(hidden_input)
    final_input = np.dot(hidden_output, W2) + b2
    final_output = final_input  # Linear activation for the output layer

    # Compute loss
    loss = mse_loss(y, final_output)

    # Backward pass
    error = mse_loss_derivative(y, final_output)
    d_W2 = np.dot(hidden_output.T, error)
    d_b2 = np.sum(error, axis=0, keepdims=True)

    error_hidden_layer = np.dot(error, W2.T) * \
        sigmoid_derivative(hidden_output)
    d_W1 = np.dot(X.T, error_hidden_layer)
    d_b1 = np.sum(error_hidden_layer, axis=0, keepdims=True)

    # Update weights and biases
    W2 -= lr * d_W2
    b2 -= lr * d_b2
    W1 -= lr * d_W1
    b1 -= lr * d_b1

    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss}')

# Demonstration of prediction
test_input = np.array([[6]])  # Predict the next number for 6
hidden_input = np.dot(test_input, W1) + b1
hidden_output = sigmoid(hidden_input)
final_input = np.dot(hidden_output, W2) + b2
prediction = final_input  # Linear activation for the output layer
print(f'Prediction for next number after 6: {prediction}')
