import numpy as np

weight = np.random.rand()
bias = np.random.randint(1,10)
input = np.random.randint(1,100)

output = np.dot(weight, input) + bias

def relu(self:output):
    return max(0,output)

print(relu(output))