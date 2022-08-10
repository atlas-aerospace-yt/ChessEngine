import chess
import random
from MachineLearning import DeepLearn
from copy import deepcopy


values={"p": -1,
        "n": -3,
        "b": -3.5,
        "r": -5,
        "q": -9,
        "k": -20,
        "P": 1,
        "N": 3,
        "B": 3.5,
        "R": 5,
        "Q": 9,
        "K": 20}

class Engine():
    def __init__(self, board=None, depth=3):
        self.board = chess.Board(board).fen() if board != None else chess.Board().fen()
        self.depth = depth
        self.positions = [[self.board]]
        self.eval = []

    def evaluate_position(self, position):
        # Defines eval
        eval = 0
        # Defines material
        material = 0
        # Splits FEN to position and gets each piece
        for piece in position.split(" ")[0]:
            # Adds value if its a piece not number or /
            material += values[piece] if piece in values else 0
        # Adds material to eval
        eval += material

        return eval

    def create_new_depth(self):
        # For each depth
        for i in range(0, self.depth):
            next_depth = []
            # Gets legal moves for each position in self.positions[-1]
            for position in self.positions[-1]:
                # Converts the fen in the list to a board
                position = chess.Board(position)
                # Calculates legal moves for every position
                legal_moves = [str(item) for item in list(position.legal_moves)]
                # Creates the next depth of positions
                for move in legal_moves:
                    # Deepcopy to prevent board being edited editing position
                    board = deepcopy(position)
                    # Makes the moves
                    board.push_san(move)
                    # Saves the new fen
                    next_depth.append(board.fen())
            # Adds the next depth to the positions
            self.positions.append(next_depth)

    def main(self):
        # Create a new depth
        self.create_new_depth()
        print(len(self.positions[-1]))

if __name__ == "__main__":
    deepLearn = DeepLearn()
    engine = Engine(depth=3)

    engine.main()

    #deepLearn.train()
