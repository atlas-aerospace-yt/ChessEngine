"""
Example use of this neural network library.
This demo has 4 examples of a vector of length 4.
Then the model trains over 750 iterations and shows the cost against epoch.
The output for 2 unseen combinations are then shown.

TODO: Overflow handling - more testing
"""

import matplotlib.pyplot as plt
from vector import Vector, activation

from model.training_methods import BackProp
from model.network import NeuralNetwork

if __name__ == "__main__":

    # Declaring examples
    example_inputs = [Vector([0,0,0,0]),
                    Vector([1,0,1,0]),
                    Vector([0,1,1,0])]

    example_outputs = [Vector(0), Vector(1), Vector(0)]

    # Initialising network
    training_method = BackProp(activation.sigmoid_prime)
    model = NeuralNetwork((4, 1, 6, 6), activation.sigmoid, training_method)

    # Training the network
    cost = model.train_network(example_inputs, example_outputs, 10000)

    # Outputs
    print(model)
    print(f"Prediction for {Vector([1,1,1,1])} is {model.forward_propagation(Vector([1,1,1,1]))}")
    print(f"Prediction for {Vector([0,1,1,1])} is {model.forward_propagation(Vector([0,1,1,1]))}")

    # Show Graph
    plt.plot(cost)
    plt.show()
