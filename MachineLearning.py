from Math import Vector as v
import __main__

class DeepLearn():

    def __init__(self):
        w, b = __main__.fileManager.load_weights_bias()
        self.move = {"w":1,"b":-1}
        self.w = v(w)
        self.b = v(b)

    def fen_to_vector(self, fen):
        # Gets the FEN position
        board = fen.split(" ")[0].split("/")
        output = []
        for row in board:
            for char in row:
                if char in __main__.values:
                    # Adds value to the piece in fen to vector
                    output.append(__main__.values[char])
                else:
                    # Adds 0 for numbers in fen
                    for i in range(0, int(char)):
                        output.append(0)
        # Adds turn to end of vector
        output.append(self.move[fen.split(" ")[1]])
        return v(output)

    def forward_propagation(self, x):
        # Forward propagation
        z = x * self.w + self.b
        return v.linear(z)

    def backward_propagation(self, x, y):
        # Forward propagation
        z = x * self.w + self.b
        y_hat = v.linear(z)

        n = len(x)

        # Optimization and backwards propagation
        w_grad = [(v.linear_prime(z) * v([x[i]]) * (2/n) * v.sum(y-y_hat))[0] for i in range(0, n)]

        self.w = self.w + v(w_grad) * 0.05

        b_grad = v.linear_prime(z) * (2/n) * v.sum(y-y_hat)
        self.b = self.b + b_grad * 0.05

    def train(self):
        # Trains the ANN
        for i in range(0, 1000):
            with open("./Data/data.txt","r") as data:
                for line in data:
                    input = self.fen_to_vector(line.split(",")[0])
                    output = v([float(line.split(",")[1])])
                    self.backward_propagation(input, output)
        # Saves the new weights and biases
        __main__.fileManager.save_weights_bias(self.w, self.b)
        # Tests the CNN
        input = self.fen_to_vector("8/8/b5k1/6pp/1B5P/6PK/8/8 w")
        print(self.forward_propagation(input))
