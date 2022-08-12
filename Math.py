import random
import sys
## TODO: ERROR HANDLING
## TODO: Commenting

E = 2.71828
PI = 3.14159265

class Vector():

    # Initialises the Vector, if the input is not a list it is turned into a list
    def __init__(self, vector):
        self.vector = vector if isinstance(vector, list) else [vector]

    # Subtraction functons
    def __sub__(self, other):
        if isinstance(other, Vector) and len(self.vector) == len(other):
            return Vector([float(self.vector[i]) - float(other[i]) for i in range(len(self))])
        elif isinstance(other, int) or isinstance(other, float):
            return Vector([(float(self[i]) - other) for i in range(len(self))])
        else:
            raise Exception(f"Error subtracting: could not subtract length {len(self.vector)} from length {len(other)}")

    # Addition function
    def __add__(self, other):
        if isinstance(other, Vector) and len(self.vector) == len(other):
            return Vector([float(self.vector[i]) + float(other[i])  for i in range(len(self))])
        elif isinstance(other, int) or isinstance(other, float):
            return Vector([float(self.vector[i]) + other  for i in range(len(self))])
        else:
            raise Exception(f"Error adding: could not add length {len(self.vector)} with length {len(other)}")

    # Multiplication function
    def __mul__(self, other):
        if isinstance(other, Vector) and len(self.vector) == len(other):
            sum = 0
            for i in range(len(self.vector)):
                sum += float(self.vector[i] * other[i])
            return Vector(sum)
        elif isinstance(other, int) or isinstance(other, float):
            return Vector([float(self.vector[i]) * other  for i in range(len(self.vector))])
        else:
            raise Exception(f"Error multiplying: could not multiply length {len(self.vector)} with length {len(other)}")

    # Returns the item if a vector is indexed e.g. myVector[2]
    def __getitem__(self, indx):
        if indx < len(self.vector) and indx >= 0:
            return self.vector[indx]
        raise Exception(f"Error: Invalid index {indx} for vector with length {len(self.vector)}")

    # Returns the length of the vector list for pythons' len(myVector) function
    def __len__(self):
        return len(self.vector)

    # Returns the vector as an easy to read string for prints or str(myVector)
    def __str__(self):
        return f"{self.vector}".replace("[","").replace("]","")

    # Returns the vectors first index as a float if float(myVector[1])
    def __float__(self):
        return self.vector[0]

    #  Returns the vectors first index as a int if int(myVector[1])
    def __int__(self):
        return self.vector[0]

    # Returns a random Vector of length x with upper and lower limits
    def random_array(x, lower=-0.1, upper=0.1):
        return Vector([random.uniform(lower, upper) for i in range(x)])

    def sigmoid(x):
        if isinstance(x, Vector):
            return Vector([1 / (1 + E ** (-x[i])) for i in range(len(x))])
        elif isinstance(x, float) or isinstance(x, int):
            return 1 / (1 + E ** (-x))

    def sigmoid_prime(x):
        return Vector([Vector.sigmoid(float(i)) * (1 - Vector.sigmoid(float(i))) for i in range(len(x))])

    def linear(x):
        if isinstance(x, Vector):
            return Vector(x * 0.5)

    def linear_prime(x):
        if isinstance(x, Vector):
            return Vector([float(0.5) for i in range(len(x))])

    def sum(x):
        sum = 0
        for item in x:
            sum += item
        return sum
