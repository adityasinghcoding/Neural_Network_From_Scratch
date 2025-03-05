# Neural Network: Forward and Back Propagation Implementation
This repository demonstrates a basic implementation of a 3-layered neural network with forward and backward propagation. It covers essential concepts of artificial neural networks, including weight updates, gradient calculations, and the use of activation functions.

## **Features**
### 1. Forward Propagation:
- Computes outputs layer by layer using matrix operations.
- Includes ReLU activation function for non-linearity.
- Three layers:
  - Input layer
  - Two hidden layers
  - Final output layer.
 
### 2. Backward Propagation:
- Calculates gradients for weights and biases using:
  - Loss function: Mean Squared Error (MSE).
  - ReLU derivative for gradient flow.
- Updates weights and biases using the learning rate.

### 3. Normalization:
Normalizes inputs and actual outputs using Min-Max scaling to ensure numerical stability during training.


# Code Highlights
## Forward Pass:
- Layer-by-layer computation with matrix multiplications.
- ReLU activation to handle non-linearity.
```
def hidden_layer(inputs, weights, bias):
  output = np.dot(inputs, weights) + bias
  return np.maximum(0, output)  # ReLU Activation
```
## Backward Pass:
- Computes weight and bias gradients for:
  - Output layer
  - Hidden layers
  - Input layer
```
def bp_layer_final(A_prev, predicted, actual):
    dL_dp = predicted - actual
    dL_dw = A_prev.T @ dL_dp
    dL_db = np.sum(dL_dp)
    return dL_dw, dL_db
```

## Weight Updates:
Updates weights and biases using gradients:
```
def update_weights(weight, bias, dL_dw, dL_db, learning_rate):
    weight = weight - learning_rate * dL_dw
    bias = bias - learning_rate * dL_db
    return weight, bias
```

## Working
### Forward Propagation:
- Input data passes through the network layer by layer.
- Outputs are computed using weights, biases, and activation functions.

## Backward Propagation:
- Calculates the gradient of the loss with respect to each weight and bias.
- Updates weights and biases using gradient descent.

## Normalization:
- Input and output normalization ensures the network trains efficiently.


# Getting Started
## Prerequisites
- Python 3.7 or higher
- NumPy for numerical computations
