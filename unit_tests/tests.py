"""
This file is designed to test features of the model and vector libraries within this repository.
"""
import os
import sys

parent = os.path.abspath('.')
sys.path.insert(1, parent)

from vector import Vector
from vector import activation

A = Vector([1, 2, 3, 4])
B = Vector([5, 6, 7, 8])

def test_multiplication():
    """
    Tests vector multiplication
    """
    assert A * B == Vector(70.0)

def test_subtraction():
    """
    Tests vector subtraction
    """
    assert B - A == Vector([4, 4, 4, 4])

def test_addition():
    """
    Tests vector addition
    """
    assert B + A == Vector([6, 8, 10, 12])

def test_sigmoid():
    """
    Tests the sigmoid function
    """
    assert activation.sigmoid(0) == 0.5

def test_int_conversion():
    """
    Tests converting a vector to an integer
    """
    assert int(Vector(1.5)) == 1

def test_float_conversion():
    """
    Tests converting a vector to a float
    """
    assert float(Vector(0.5)) == 0.5

def test_vector_summation():
    """
    Tests summing a vector
    """
    assert sum(A) == 10
