"""
This is an example file to test image recognition using this neural network
"""

from PIL import Image

from model.network import NeuralNetwork
from model.training_methods import BackProp
from vector import Vector, activation


def image_to_vector(path):
    """
    converts image to vector

    Args:
        path (string): path to the file

    Returns:
        Vector: the image in vector form
    """
    # gets image, resizes then converts to greyscale
    image = Image.open(path)
    image = image.resize((100,100))
    grey_image = image.convert('L')

    # now the image must be converted to an image
    pixels = list(grey_image.getdata())
    width, height = grey_image.size
    pixels = [pixels[i * width:(i + 1) * width] for i in range(height)]

    return Vector([item / 255 for item in [j for sub in pixels for j in sub]])

if __name__ == "__main__":

    examples = ["perry1.png", "perry2.png", "notperry.png"]
    results = [1, 1, 0]

    training_method = BackProp(activation.sigmoid_prime)

    EPOCH = 2500

    cost = []

    network = NeuralNetwork((10000, 1, 1, 1)
                            , activation.sigmoid, training_method)

    network.learn_rate = 1

    image_vectors = [image_to_vector(image) for image in examples]

    result_vectors = [Vector(result) for result in results]

    for i in range(EPOCH):
        network.train_network(image_vectors[0], result_vectors[0])
        network.train_network(image_vectors[1], result_vectors[1])
        network.train_network(image_vectors[2], result_vectors[2])

        print(i)

    print(network.forward_propagation(image_to_vector(examples[0])))
    print(network.forward_propagation(image_to_vector(examples[1])))
    print(network.forward_propagation(image_to_vector(examples[2])))
    print(network.forward_propagation(image_to_vector("unknown.png")))
