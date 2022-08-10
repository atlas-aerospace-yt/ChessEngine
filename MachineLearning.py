from timeit import default_timer as timer
from Math import Vector as v
import numpy as np
from FileManager import FileManager

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
        self.weightOne = [v.random_array(5, lower=-1, upper=1) for i in range(0, 200)]
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

    def forward(self, input):
        one = v.sigmoid(v([input * node for node in self.weightOne]) + self.biasOne)
        two = v.sigmoid(v([one * node for node in self.weightTwo]) + self.biasTwo)
        three = v.sigmoid(v([two * node for node in self.weightThree]) + self.biasThree)
        return v.sigmoid(v([three * node for node in self.weightFour]) + self.biasFour)

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

    def backward(self, input, output):

        alpha = 0.05

        one = v.sigmoid(v([input * node for node in self.weightOne]) + self.biasOne)
        two = v.sigmoid(v([one * node for node in self.weightTwo]) + self.biasTwo)
        three = v.sigmoid(v([two * node for node in self.weightThree]) + self.biasThree)
        four = v.sigmoid(v([three * node for node in self.weightFour]) + self.biasFour)

        C_aL = (four - output) * (2 * 1 / 5)

        wFourGrad, bFourGrad, C_aLOne = self.weightGradient_biasGradient_outputGradient(C_aL, v.sigmoid_prime(four), three, self.weightFour)
        wThreeGrad, bThreeGrad, C_aLTwo = self.weightGradient_biasGradient_outputGradient(C_aLOne, v.sigmoid_prime(three), two, self.weightThree)
        wTwoGrad, bTwoGrad, C_aLThree = self.weightGradient_biasGradient_outputGradient(C_aLTwo, v.sigmoid_prime(two), one, self.weightTwo)
        wOneGrad, bOneGrad = self.weightGradient_biasGradient_outputGradient(C_aLThree, v.sigmoid_prime(one), input)

        self.weightFour = [self.weightFour[i] - wFourGrad[i] * alpha for i in range(len(self.weightFour))]
        self.weightThree = [self.weightThree[i] - wThreeGrad[i] * alpha for i in range(len(self.weightThree))]
        self.weightTwo = [self.weightTwo[i] - wTwoGrad[i] * alpha for i in range(len(self.weightTwo))]
        self.weightOne = [self.weightOne[i] - wOneGrad[i] * alpha for i in range(len(self.weightOne))]

        self.biasFour = self.biasFour - bFourGrad * alpha
        self.biasThree = self.biasThree - bThreeGrad * alpha
        self.biasTwo = self.biasTwo - bTwoGrad * alpha
        self.biasOne = self.biasOne - bOneGrad * alpha


# Example code if you would like to run MachineLearning.py alone
# The ANN learns to recognise the only importance is the fourth indx
# You can edit the inputs and outputs for it to learn other patterns
if __name__ == "__main__":
    learning = DeepLearn()

    for i in range(500):
        print(i)
        learning.backward(v([0,0,0,0,1]), v([1]))
        learning.backward(v([0,1,0,0,1]), v([1]))
        learning.backward(v([1,0,0,0,1]), v([1]))
        learning.backward(v([0,0,1,0,1]), v([1]))
        learning.backward(v([0,1,0,0,1]), v([1]))
        learning.backward(v([1,0,0,0,0]), v([0]))
        learning.backward(v([0,1,0,0,0]), v([0]))
        learning.backward(v([0,0,0,1,0]), v([0]))
        learning.backward(v([1,0,1,1,0]), v([0]))

    learning.save_layers()

    print(learning.forward(v([1,1,1,1,0])))
    print(learning.forward(v([0,0,0,0,1])))
