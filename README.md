# ChessEngine
A python chess engine.

## Contents

[The Project](#project)

<a name="project"/>
## The Project
Currently the code is under development in the most early of stages.</br>
The Machine Learning library is a simple one layer network, however this will be changed hopefully to a 6-8 layer DNN (Deep Neural Network) to allow for better pattern recognition which is needed for a game like chess.</br>

<a name="minmax"/>
## MinMax Algorithm
This is an algorithm that works by first calculating each possible position. Then you evaluate each position with some sort of evaluation function (in this case we use the Neural Network). Now you go through the tree choosing the best options for you and the best options with the opponent.</br>
In chess, this uses a huge amount of data, number of possible boards go: 20, 400, 8902...</br>
To get around this you can use methods such as alpha-beta.

<a name="eval"/>
## Evaluation
To evaluate chess positions, this code is programmed to use an ANN (Artificial Neural Network) which trains off of an analysis from the currently strongest open source chess engine [Stockfish](https://stockfishchess.org/).

<a name="math"/>
## Math Library
I created a simple vector library based around neural network maths. Numpy is a great mathematics library however lacked a few functions that are needed for ANNs. 

<a name="directory"/>
## Directory Structure
```
Engine
|
|-- ANN                 # Weights and biases are stored here.
|
|-- DaTa                # Training data goes here.
|
|-- Engine.py           # Main file, holds the engine class.
|-- FileManager.py      # Controls files, saves training data and ANN data.
|-- MachineLearning.py  # Contains an ANN which evaluates position from fen.
|-- Math.py             # A custom library which does vector math.

```
<a name="dependencies"/>
## Dependencies
```
pip install chess
```
