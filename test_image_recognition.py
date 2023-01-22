"""
An example use for the image wrapper in the model library
"""

import os

from model.image import ImageRecognition

recogniser = ImageRecognition(f"{os.getcwd()}\\example_data")
recogniser.learn_images(epoch=2000)

print(recogniser.recognise(
    f"{os.getcwd()}\\example_data\\perry\\perry1.png", show=True))
print(recogniser.recognise(f"{os.getcwd()}\\example_data\\not_perry\\notperry1.png", show=True))
