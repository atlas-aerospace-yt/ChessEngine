import chess
import random
from MachineLearning import DeepLearn
from copy import deepcopy

## TODO: finish MinMax
## TODO: reinforcement learning algorithm

class Engine():

    def __init__(self):
        pass

    def evaluate_position(self, position):
        # Converts the fen position to a vector
        input_vector = deepLearn.fen_to_vector(position)
        # Gets the value from the MachineLearning library
        eval = deepLearn.predict(input_vector)
        return eval

    def min_max(self, depth, fen):
        #[
        #  [[position 1]]
        #  [[position 1, position2]]
        #  [[position 1, position 2], [position 1, position 2]]
        #  [[position 1, position 2], [position 1, position 2],[position 1, position 2], [position 1, position 2]]
        #]
        positions_tree = [[[fen]]]
        for i in range(depth):
            next_depth = []
            position_numbers = 0
            for position_list in positions_tree[-1]:
                for position in position_list:
                    position = chess.Board(position)
                    legal_moves = [str(item) for item in list(position.legal_moves)]
                    sub_pos_list = []
                    for move in legal_moves:
                        board = deepcopy(position)
                        board.push_san(move)
                        sub_pos_list.append(board.fen())
                    next_depth.append(sub_pos_list)
            positions_tree.append(next_depth)
        print(len(positions_tree[-1]))

        #print(next_depth)
        #print(len(next_depth))

        #eval_tree = [[self.evaluate_position(position) for position in position_list] for position_list in positions_tree[-1]]
        #max = False
        #prev_layer = []
        #for eval_list in eval_tree:
        #    if max:
        #        prev_layer.append(max(eval_list))
        #    else:
        #        prev_layer.append(min(eval_list))
        #
        #print(prev_layer)


    def main(self):
        # Defines the starting position
        board = chess.Board()
        self.min_max(4, str(board.fen()))
        # A loop to play a game of chess
        while True:
            move = [str(item) for item in list(board.legal_moves)]

            if len(move) == 0 or board.is_checkmate() or board.is_stalemate() or board.is_insufficient_material():
                board = chess.Board()
                if board.is_stalemate() or board.is_insufficient_material():
                    print(f"Game Over: draw")
                    return 0
                else:
                    print(f"Game Over: {board.turn} wins")
                    return 0
            else:
                if board.turn:
                    move = input(f"{[str(item) for item in list(board.legal_moves)]}\nPlease enter a legal move: ")
                    board.push_san(move)
                else:
                    #board.push_san(move[random.randint(0, len(move)-1)])
                    #data = f"{board.fen()} {self.evaluate_position(board.fen())} \n"
                    #fileManager.save_training_data(data)
                    print(board.fen())
                    self.min_max(3, str(board.fen()))
                print(f"{board.turn}\n{board}")

if __name__ == "__main__":
    deepLearn = DeepLearn()
    engine = Engine()

    engine.main()
