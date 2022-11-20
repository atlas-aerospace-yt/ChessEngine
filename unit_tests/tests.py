"""
This file is designed to test features of the model and vector libraries within this repository
"""
import os
import sys

parent = os.path.abspath('.')
sys.path.insert(1, parent)

from vector import Vector
from vector import activation

A = Vector([1,2,3,4])
B = Vector([5,6,7,8])

def test_multiplication():
    """
    Tests vector multiplication
    """
    assert float(A * B) == 70.0

def test_sigmoid():
    """
    Tests the sigmoid function
    """
    assert activation.sigmoid(0) == 0.5