from timeit import default_timer as timer
from FileManager import FileManager
from Math import Vector as v
import numpy as np

class DeepLearn():
    def __init__(self):
        # Defines the file manager
        self.fileManager = FileManager()
        # Generates new network if size changes
        if self.fileManager.get_num_of_layers() != 4:
            self.generate_new_network()
            return None

        self.weightOne, self.biasOne = self.fileManager.load_weights_bias(1)
        self.weightTwo, self.biasTwo = self.fileManager.load_weights_bias(2)
        self.weightThree, self.biasThree = self.fileManager.load_weights_bias(3)
        self.weightFour, self.biasFour = self.fileManager.load_weights_bias(4)

    def generate_new_network(self):

        print("Starting new network")
        # Cleares if new network size is defined
        self.fileManager.clear_ann_data()

        # Initialises and saves weights
        self.weightOne = [v.random_array(69, lower=-1, upper=1) for i in range(0, 200)]
        self.biasOne = v.random_array(200, lower=-1, upper=1)

        self.weightTwo = [v.random_array(200, lower=-1, upper=1) for i in range(0, 200)]
        self.biasTwo = v.random_array(200, lower=-1, upper=1)

        self.weightThree = [v.random_array(200, lower=-1, upper=1) for i in range(0, 200)]
        self.biasThree = v.random_array(200, lower=-1, upper=1)

        self.weightFour = [v.random_array(200, lower=-1, upper=1) for i in range(0, 1)]
        self.biasFour = v.random_array(1, lower=-1, upper=1)

        self.save_layers()

    def save_layers(self):
        self.fileManager.save_weights_bias(self.weightOne, self.biasOne, 1)
        self.fileManager.save_weights_bias(self.weightTwo, self.biasTwo, 2)
        self.fileManager.save_weights_bias(self.weightThree, self.biasThree, 3)
        self.fileManager.save_weights_bias(self.weightFour, self.biasFour, 4)

    def predict(self, input):
        one = v.sigmoid(v([input * node for node in self.weightOne]) + self.biasOne)
        two = v.sigmoid(v([one * node for node in self.weightTwo]) + self.biasTwo)
        three = v.sigmoid(v([two * node for node in self.weightThree]) + self.biasThree)
        return float(v.sigmoid(v([three * node for node in self.weightFour]) + self.biasFour))

    def train(self):

        for i in range(50):
            alpha = 0.05

            one = v.sigmoid(v([input * node for node in self.weightOne]) + self.biasOne)
            two = v.sigmoid(v([one * node for node in self.weightTwo]) + self.biasTwo)
            three = v.sigmoid(v([two * node for node in self.weightThree]) + self.biasThree)
            four = v.sigmoid(v([three * node for node in self.weightFour]) + self.biasFour)

            C_Four = (four - output) * (2 * 1 / 5)

            wFourGrad, bFourGrad, C_Three = self.weightGradient_biasGradient_outputGradient(C_Four, v.sigmoid_prime(four), three, self.weightFour)
            wThreeGrad, bThreeGrad, C_Two = self.weightGradient_biasGradient_outputGradient(C_Three, v.sigmoid_prime(three), two, self.weightThree)
            wTwoGrad, bTwoGrad, C_One = self.weightGradient_biasGradient_outputGradient(C_Two, v.sigmoid_prime(two), one, self.weightTwo)
            wOneGrad, bOneGrad = self.weightGradient_biasGradient_outputGradient(C_One, v.sigmoid_prime(one), input)

            self.weightFour = [self.weightFour[i] - wFourGrad[i] * alpha for i in range(len(self.weightFour))]
            self.weightThree = [self.weightThree[i] - wThreeGrad[i] * alpha for i in range(len(self.weightThree))]
            self.weightTwo = [self.weightTwo[i] - wTwoGrad[i] * alpha for i in range(len(self.weightTwo))]
            self.weightOne = [self.weightOne[i] - wOneGrad[i] * alpha for i in range(len(self.weightOne))]

            self.biasFour = self.biasFour - bFourGrad * alpha
            self.biasThree = self.biasThree - bThreeGrad * alpha
            self.biasTwo = self.biasTwo - bTwoGrad * alpha
            self.biasOne = self.biasOne - bOneGrad * alpha

    def fen_to_vector(self, fen):
        # Dictionary definitions
        values={"p": -1,"n": -3,"b": -3.5,"r": -5,"q": -9,"k": -20,"P": 1,"N": 3,"B": 3.5,"R": 5,"Q": 9,"K": 20}
        turn={"w":1,"b":-1}
        # Gets the FEN position
        board = fen.split(" ")[0].split("/")
        output = []
        for row in board:
            for char in row:
                if char in values:
                    # Adds value to the piece in fen to vector
                    output.append(values[char])
                else:
                    # Adds 0 for numbers in fen
                    for i in range(0, int(char)):
                        output.append(0)
        # Adds turn to end of vector
        output.append(turn[fen.split(" ")[1]])
        # Adds castling rights
        if fen.split(" ")[2] != "-" and not fen.split(" ")[2].isnumeric():
            output.append(1) if "K" in fen.split(" ")[2] else output.append(0)
            output.append(1) if "Q" in fen.split(" ")[2] else output.append(0)
            output.append(1) if "k" in fen.split(" ")[2] else output.append(0)
            output.append(1) if "q" in fen.split(" ")[2] else output.append(0)
        return v(output)

    def weightGradient_biasGradient_outputGradient(self, C_aL, aL_zL, zL_wL, zL_aLOne=None):
        weightGradient = []
        for n in range(len(C_aL)):
            weightGradient.append(v([C_aL[n] * aL_zL[n] * zL_wL[i] for i in range(len(zL_wL))]))
        biasGradient = v([C_aL[n] * aL_zL[n] for n in range(len(C_aL))])
        if zL_aLOne != None:
            outputGradient = []
            for k in range(len(zL_aLOne[0])):
                sum = 0
                for j in range(len(C_aL)):
                    sum += C_aL[j] * aL_zL[j] * zL_aLOne[j][k]
                outputGradient.append(sum)
            return weightGradient, biasGradient, outputGradient
        return weightGradient, biasGradient
