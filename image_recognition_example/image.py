"""
This is an example file to test image recognition using this neural network

To run this file:
    1) navigate to the "ChessEngine" directory
    2) run the command python ./ChessEngine/image.py

This file must be run from the ChessEngine directory
"""

from PIL import Image
import os
import sys
import matplotlib.pyplot as plt

parent = os.path.abspath('.')
sys.path.insert(1, parent)

from model.training_methods import BackProp
from model.network import NeuralNetwork

from vector import Vector, activation

def image_to_vector(path, show=False):
    """
    converts image to vector

    Args:
        path (string): path to the file
        show (bool): should the images be displayed
    Returns:
        Vector: the image in vector form
    """

    # gets image, resizes then converts to greyscale
    image = Image.open(path)
    image = image.resize((32,32))
    grey_image = image.convert('L')

    # displays the images
    if show:
        grey_image.show()

    # now the image must be converted to an image
    pixels = list(grey_image.getdata())
    width, height = grey_image.size
    pixels = [pixels[i * width:(i + 1) * width] for i in range(height)]

    return Vector([item / 255 for item in [j for sub in pixels for j in sub]])

if __name__ == "__main__":
    # example definitions
    examples = ["perry1.png", "perry2.png", "notperry1.png"]
    results = [1, 1, 0]

    # converts images to list
    image_vectors = [image_to_vector(
f"./image_recognition_example/examples/{image}") for image in examples]
    result_vectors = [Vector(result) for result in results]

    # Initialising network
    training_method = BackProp(activation.sigmoid_prime)
    model = NeuralNetwork((1024, 1, 3, 10), activation.sigmoid, training_method)
    model.training_method.learn_rate = 0.5

    # Training the network
    cost = model.train_network(image_vectors, result_vectors, 200)
    
    print(model)
    unknown = image_to_vector("./image_recognition_example/tests/unknown.png", show=True)
    print(model.forward_propagation(unknown))

    plt.plot(cost)
    plt.show()