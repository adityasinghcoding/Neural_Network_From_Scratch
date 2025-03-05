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
```def hidden_layer(inputs, weights, bias):
    output = np.dot(inputs, weights) + bias
    return np.maximum(0, output)  # ReLU Activation
```
