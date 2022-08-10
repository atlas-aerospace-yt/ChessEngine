import os
import glob
import __main__
from Math import Vector as v

class FileManager():

    def __init__(self):
        # Creates the directories to save the neural networks data
        if not os.path.exists("./ANN"): os.makedirs("./ANN")
        # Creates the traning data files for the ANN
        if not os.path.exists("./Data"): os.makedirs("./Data")

    def save_weights_bias(self, w, b, layer):
        # Saves the string form of the vector to a .txt file
        with open(f"./ANN/{layer}weights.txt","w") as weight_file:
            for n in w:
                weight_file.write(f"{str(n)}\n")
        with open(f"./ANN/{layer}bias.txt","w") as bias_file:
            bias_file.write(str(b))

    def load_weights_bias(self, layer):
        # Reads the vector and converts them into numeric lists
        with open(f"./ANN/{layer}weights.txt","r") as weight_file:
            weights = []
            for line in weight_file:
                weights.append(v([float(item) for item in line.split(", ")]))
        with open(f"./ANN/{layer}bias.txt","r") as bias_file:
            for line in bias_file:
                bias = v([float(item) for item in line.split(", ")])
        # Returns the weights and biases
        return weights, bias

    def save_training_data(self, data):
        # Saves training data to the data.txt in the form of "FEN turn,eval"
        with open("./Data/data.txt","a") as save:
            save.write(data)

    def clear_ann_data(self):
        # Clears the saved data about the ANN
        for file in os.listdir("./ANN"):
            os.remove(f"./ANN/{file}")

    def get_num_of_layers(self):
        # Returns the number of layers that are saved
        return len(os.listdir("./ANN")) / 2
