from .board import Board
import numpy as np


class Environment:

    def __init__(self):
        self.board = Board()
        self.current_player = None

    def reset(self, cur):
        self.board.reset()
        assert cur == 1 or cur == -1
        self.current_player = cur
        return self.state

    @property
    def state(self):
        if self.current_player == 1:
            return np.array([self.board.valid, self.board.positive, self.board.negative], dtype=np.float32)
        else:
            return np.array([self.board.valid, self.board.negative, self.board.positive], dtype=np.float32)

    def sample(self):
        return self.board.sample()

    def step(self, action):
        valid = self.board.get_valid()
        if action not in valid:
            r = -1
        else:
            r = .5
        game_over, value = self.board.put(action, self.current_player)
        if not game_over:
            game_over, value = self.board.check_game_over(self.current_player)
        self.current_player *= -1
        return self.state, r, game_over, value

    def print_board(self):
        print('-------------')
        for i in range(3):
            print('|', end='')
            for j in self.board.field[i]:
                if j == 1:
                    s = '○'
                elif j == -1:
                    s = 'x'
                else:
                    s = ' '
                print(f' {s} |', end='')
            print('')
            print('-------------')
