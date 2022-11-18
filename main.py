"""
main file to test the libraries while under development
"""

from machine_learning import Layer
from vector import activation
from vector import Vector

layer = Layer(10, 1, activation.sigmoid)

print(layer)

a = Vector([1, 2, 3, 4])
b = Vector([5, 6, 7, 8])

if float(a * b) != 70.0:
    print("Error: multiplication failed!")
