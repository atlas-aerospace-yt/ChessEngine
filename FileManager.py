import os
import __main__

class FileManager():

    def __init__(self):
        # Creates the directories to save the neural networks data
        if not os.path.exists("./ANN"): os.makedirs("./ANN")
        # Creates the .txt files
        file = open("./ANN/weights.txt","a")
        file.close()
        file = open("./ANN/bias.txt","a")
        file.close()
        # Creates the traning data files for the ANN
        if not os.path.exists("./Data"): os.makedirs("./Data")

    def save_weights_bias(self, w, b):
        # Saves the string form of the vector to a .txt file
        with open("./ANN/weights.txt","w") as weight_file:
            weight_file.write(str(w))
        with open("./ANN/bias.txt","w") as bias_file:
            bias_file.write(str(b))

    def load_weights_bias(self):
        # Reads the vector and converts them into numeric lists
        with open("./ANN/weights.txt","r") as weight_file:
            for line in weight_file:
                weights = [float(item) for item in line.split(", ")]
        with open("./ANN/bias.txt","r") as bias_file:
            for line in bias_file:
                bias = [float(item) for item in line.split(", ")]
        # Returns the weights and biases
        return weights, bias

    def save_training_data(self, data):
        # Saves training data to the data.txt in the form of "FEN turn,eval"
        with open("./Data/data.txt","a") as save:
            save.write(data)
