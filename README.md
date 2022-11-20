![pylint](https://github.com/atlas-aerospace-yt/ChessEngine/actions/workflows/pylint.yml/badge.svg) ![pylint](https://github.com/atlas-aerospace-yt/ChessEngine/actions/workflows/python-app.yml/badge.svg)

<p align="center"><img src=images/style.png height=170 width=300></img></p>

<h1 align="center">Chess Engine</h1>

<h4 align="center"> 
DISCLAIMER</br>
This is an on-going project which is not complete and in very early stages of development.
</h4>

## Contents

[The Project](#the-project)</br>
[Vector Library](#vector-library)</br>
[Model Library](#model-library)</br>
[MinMax Algorithm](#minmax-algorithm)</br>
[Evaluation](#evaluation)</br>

## The Project

Currently the code is under heavy development in the most early of stages.</br>
The Machine Learning library is a simple one layer network, however this will be changed hopefully to a 6-8 layer DNN (Deep Neural Network) to allow for better pattern recognition which is needed for a game like chess.</br>
The MinMax code has also not been started yet. However, I would like to include alpha beta pruning as currently, the code is very slow taking a long time to see just four moves deep.</br>
The FileManger is just to keep track of files so files aren't being modified in separate places and is easy to read - There will soon be a PseudoData file added as I will be moving to a semi-supervised algorithm.</br>
The math library is one of my favourite pieces of code as it makes the Neural Network maths easy...

## Vector Library

```
Vector structure:

vector
|
|-- __init__.py     # initialisation file
|   |
|   |- Vector       # vector class
|
|-- activation.py   # holds activation functions for vector
|
|-- random.py       # holds the generate random vector function
|
```

The Vector object can be imported by 'from vector import Vector' then can be used.

To generate a random vector `from vector import random` then you can `my_vector = random.random_vector(length_of_vector)`.

The activation function holds 4 functions: `sigmoid`, `sigmoig_prime`, `linear`, `linear_prime`. They all take the input of one Vector object.

## Model Library

```
Model structure:

Model
|
|-- __init__.py         # initialisation file
|
|-- layer.py
|   |
|   |- Layer            # layer class
|
|-- network.py
|   |
|   |- NeuralNetwork    # neural network clas
|
```

To create a neural network import `from model.network import NeuralNetwork`. To initialise pass in `num_of_input, num_of_output, num_of_layers, num_of_nodes, activation_func`.
This class holds the `cost_function` which can be used to calculate the cost of the network. To predict something based on the input, use the `forward_propagation` which takes one vector as the input.

The network holds a list of Layers which you shouldn't need to touch.

## MinMax Algorithm

This is an algorithm that works by first calculating each possible position. Then you evaluate each position with some sort of evaluation function (in this case we use the Neural Network). Now you go through the tree choosing the best options for you and the best options with the opponent.</br>
In chess, this uses a huge amount of data, number of possible boards go: 20, 400, 8902...</br>
To get around this you can use methods such as alpha-beta.

## Evaluation

To evaluate chess positions, this code is programmed to use an ANN (Artificial Neural Network) which trains off of an analysis from the currently strongest open source chess engine [Stockfish](https://stockfishchess.org/).</br>
The goal for the evaluation is to have it recognise drawing positions such as opposite coloured bishop endgames or king Vs king and pawn endgames and also for it to recognise when there is a winning player either positionally or by material.</br>
This will also be changed to a semi-supervised learning algorithm where the code trains off of the data in `./Data/` then generates pseudo labels for unlabelled data to further train off.
