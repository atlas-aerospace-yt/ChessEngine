"""
A python maths library oriented around machine learning

Contains:

class Vector

function linear
function linear_prime
function sigmoid
function sigmoig_prime
function random_array
function sum_vector

TODO: ERROR HANDLING
"""

import random


E = 2.71828
PI = 3.14159265


class Vector():
    """
    Vector class

    Designed to create a vector object to be used by the main program
    """

    def __init__(self, vector):
        self.vector = vector if isinstance(vector, list) else [vector]

    def __sub__(self, other):
        if isinstance(other, Vector) and len(self.vector) == len(other):
            return Vector([float(self.vector[i]) - float(other[i]) for i in range(len(self))])
        if isinstance(other, int) or isinstance(other, float):
            return Vector([(float(self[i]) - other) for i in range(len(self))])

        raise Exception(f"Error subtracting: \
could not subtract length {len(self.vector)} from length {len(other)}")

    def __add__(self, other):
        if isinstance(other, Vector) and len(self.vector) == len(other):
            return Vector([float(self.vector[i]) + float(other[i]) for i in range(len(self))])
        if isinstance(other, int) or isinstance(other, float):
            return Vector([float(self.vector[i]) + other for i in range(len(self))])

        raise Exception(f"Error adding: \
could not add length {len(self.vector)} with length {len(other)}")

    def __mul__(self, other):
        if isinstance(other, Vector) and len(self.vector) == len(other):
            sum_ans = 0
            for index, value in enumerate(self.vector):
                sum_ans += float(value * other[index])
            return Vector(sum_ans)
        if isinstance(other, int) or isinstance(other, float):
            return Vector([float(self.vector[i]) * other for i in range(len(self.vector))])

        raise Exception(f"Error multiplying: \
could not multiply length {len(self.vector)} with length {len(other)}")

    def __getitem__(self, index):
        if index < len(self.vector) and index >= 0:
            return self.vector[index]

        raise Exception(f"Error: \
Invalid index {index} for vector with length {len(self.vector)}")

    def __len__(self):
        return len(self.vector)

    def __str__(self):
        return f"{self.vector}".replace("[", "").replace("]", "")

    def __float__(self):
        return self.vector[0]

    def __int__(self):
        return self.vector[0]


def random_vector(num, lower=-0.1, upper=0.1):
    """
    Generates a random array

    Args:
        num (int): length of the array
        lower (float): minimum value in the vector
        upper (float): maximum value in the vector
    Return:
        Vector object
    """
    return Vector([random.uniform(lower, upper) for i in range(num)])


def sigmoid(num):
    """
    Applies the sigmoid function to the input

    Args:
        num (int or float or Vector): The object to apply the sigmoid curve to
    Return:
        int or float or Vector: the result of the sigmoid funciton
    """
    if isinstance(num, Vector):
        return Vector([1 / (1 + E ** (-num[i])) for i in range(len(num))])
    elif isinstance(num, float) or isinstance(num, int):
        return 1 / (1 + E ** (-num))
    else:
        return 0


def sigmoid_prime(num):
    """
    Applies the sigmoid prime function to the input

    Args:
        num (int or float or Vector): The object to apply the sigmoid prime curve to
    Return:
        int or float or Vector: the result of the sigmoid prime funciton
    """
    temp_list = []

    for value in num:
        result = sigmoid(value) * (1-sigmoid(value))
        temp_list.append(result)

    return Vector(temp_list)


def linear(num):
    """
    Linear actvation function y=kx

    Args:
        num (int or float or Vector): The object to apply the weight to
    Return:
        int or float or Vector: the result of the weight
    """
    if isinstance(Vector):
        return Vector(num * 0.5)
    else:
        return num * 0.5


def linear_prime(num):
    """
    Linear actvation function y=kx

    Args:
        num (int or float or Vector): The object to apply the derivative to
    Return:
        int or float or Vector: the result of the derivative
    """
    if isinstance(num, Vector):
        return Vector([float(0.5) for i in range(len(num))])
    else:
        return 0.5


def sum_vector(num):
    """
    Summates every item within the vector

    Args:
        num (Vector): the argument to summate the items within
    Return:
        int: the sum of all items within the list
    """
    sum_vector_float = 0
    for item in num:
        sum_vector_float += item
    return sum_vector_float
