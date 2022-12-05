"""
A wrapper file for machine learning image recognition
"""

import os

from PIL import Image

from vector import Vector, activation

from model.network import NeuralNetwork
from model.training_methods import BackProp

class ImageRecognition:
    """
    Wrapper class for machine learning image recognition
    """

    def __init__(self, training_directory):

        training_method = BackProp(activation.sigmoid_prime)
        number_of_outputs = len(os.listdir(training_directory))
        network_stats = (1024, number_of_outputs, 10, 10)

        self.model = NeuralNetwork(network_stats, activation.sigmoid, training_method)
        self.training_dir = training_directory

    def recognise_image(self, show=False):
        """
        trains the neural network to recognise images
        """

        training_folders = os.listdir(self.training_dir)

        for training_img in training_folders:
            print(training_img, os.listdir(f"./{self.training_dir}\\\"{training_img}\""))

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
