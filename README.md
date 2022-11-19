![pylint](https://github.com/atlas-aerospace-yt/ChessEngine/actions/workflows/pylint.yml/badge.svg)

<p align="center"><img src=images/style.png height=270 width=480></img></p>

<h1 align="center">Chess Engine</h1>

<h4 align="center"> DISCLAIMER: This is an on-going project which is not complete. </h4>

## Contents

[The Project](#the-project)</br>
[Vector Library](#vector-library)</br>
[MinMax Algorithm](#minmax-algorithm)</br>
[Evaluation](#evaluation)</br>
[Directory Structure](#directory-structure)</br>

## The Project

Currently the code is under heavy development in the most early of stages.</br>
The Machine Learning library is a simple one layer network, however this will be changed hopefully to a 6-8 layer DNN (Deep Neural Network) to allow for better pattern recognition which is needed for a game like chess.</br>
The MinMax code has also not been started yet. However, I would like to include alpha beta pruning as currently, the code is very slow taking a long time to see just four moves deep.</br>
The FileManger is just to keep track of files so files aren't being modified in separate places and is easy to read - There will soon be a PseudoData file added as I will be moving to a semi-supervised algorithm.</br>
The math library is one of my favourite pieces of code as it makes the Neural Network maths easy...

## Vector Library

This is a library which holds a vector object and holds functions to perform machine learning.

`class Vector` is a class which holds a single object `self.vector` which is a list. This class takes one object for initialisation.

The rest of the functions within the class are built-in python operators such as `__mul__` or `__getitem__`.

To generate a random vector this file holds `def random_vector(length, lower=-0.1, upper=0.1)`. `length` is the length of the random vector and `lower` and `upper` are the lower and upper bounds respectively.

The library also holds `sigmoid`, `sigmoid_prime`, `linear`, and `linear_prime` which can be used as activation functions. They accept int, float, or vector.

To summate the whole vector `sum_vector` is used which takes the input of one vector and returns an int.

## MinMax Algorithm

This is an algorithm that works by first calculating each possible position. Then you evaluate each position with some sort of evaluation function (in this case we use the Neural Network). Now you go through the tree choosing the best options for you and the best options with the opponent.</br>
In chess, this uses a huge amount of data, number of possible boards go: 20, 400, 8902...</br>
To get around this you can use methods such as alpha-beta.

## Evaluation

To evaluate chess positions, this code is programmed to use an ANN (Artificial Neural Network) which trains off of an analysis from the currently strongest open source chess engine [Stockfish](https://stockfishchess.org/).</br>
The goal for the evaluation is to have it recognise drawing positions such as opposite coloured bishop endgames or king Vs king and pawn endgames and also for it to recognise when there is a winning player either positionally or by material.</br>
This will also be changed to a semi-supervised learning algorithm where the code trains off of the data in `./Data/` then generates pseudo labels for unlabelled data to further train off.

## Directory Structure
```
Engine
|
|-- ANN                 # Weights and biases are stored here.
|
|-- Data                # Training data goes here.
|
|-- Engine.py           # Main file, holds the engine class.
|-- FileManager.py      # Controls files, saves training data and ANN data.
|-- MachineLearning.py  # Contains an ANN which evaluates position from fen.
|-- Math.py             # A custom library which does vector math.

```
