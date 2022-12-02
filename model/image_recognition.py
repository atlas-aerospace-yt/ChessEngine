"""
TODO

A wrapper file for setting up a model and training it
to recognise images.

Dependencised:
    PIL
    matplotlib
    vector
"""

import os
from PIL import Image

from model.network import NeuralNetwork
from model.training_methods import BackProp
from vector import Vector, activation

class ImgRecognition():
    """
    TODO

    Holds wrapper functions to perform image recognition.

    The network will initialise with 1024 - > 20 -> num_of_categories

    Args:
        example_dir (str): the string of the directory of images
        NOTE structure:
            example_dir
            |
            | - grouped images 1 (e.g. dog)
            |
            | - grouped images 2 (e.g. cat)
            |
            | - grouped images 3 (e.g. neither)

    Attributes:
        network (NeuralNetwork): the neural net to train for images
    """

    def __init__(self, example_dir):

        files = os.listdir(example_dir)
        num_of_outputs = len(files)

        network_stats = (1024, 3, 20, num_of_outputs)
        training_method = BackProp(activation.sigmoid_prime)
        self.network = NeuralNetwork(network_stats, activation.sigmoid, training_method)

    def recognise(self):
        """
        Recognises patterns within the images
        """
        pass

    def learn_images(self):
        """
        Trains the network to recognise images
        """
        pass

    def image_to_vector(self, path, show=False):
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
