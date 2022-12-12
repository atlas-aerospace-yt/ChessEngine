"""
A wrapper file for machine learning image recognition
"""

import os

import matplotlib.pyplot as plt

from PIL import Image

from vector import Vector, activation

from model.network import NeuralNetwork
from model.training_methods import BackProp

class ImageRecognition:
    """
    Wrapper class for machine learning image recognition
    """

    def __init__(self, training_path):

        training_method = BackProp(activation.sigmoid_prime)
        self.number_of_outputs = len(os.listdir(training_path))
        network_stats = (1024, self.number_of_outputs, 1, 1)

        self.model = NeuralNetwork(network_stats, activation.sigmoid, training_method)
        self.training_path = training_path

    def recognise(self, path):
        """
        Recognises the image based off of training data

        Args:
            path (str): the path to the image

        Returns:
            str: the name of the file that the data predicts
        """

        # Gets the output of the network
        out = list(self.model.forward_propagation(self.image_to_vector(path)))

        # Converts the output to the name of the file
        return os.listdir(self.training_path)[out.index(max(out))]

    def learn_images(self, show=False):
        """
        trains the neural network to recognise images
        """

        training_folders = os.listdir(self.training_path)

        examples = []
        outputs = []

        # gets the exampels and expected outputs
        for index, training_img_dir in enumerate(training_folders):

            initial_len = len(examples)

            # computes the expected output for the file
            output = [0 for i in range(self.number_of_outputs)]
            output[index] = 1
            output = Vector(output)

            for file in os.listdir(f"{self.training_path}\\{training_img_dir}"):
                examples.append(self.image_to_vector(
                    f"{self.training_path}\\{training_img_dir}\\{file}"))

            for _ in range(len(examples) - initial_len):
                outputs.append(output)

        # trains the network
        cost = self.model.train_network(examples, outputs, 5000)

        # shows the cost function
        if show:
            plt.plot(cost)
            plt.show()

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
