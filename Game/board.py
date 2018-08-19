import numpy as np


class Board:

    def __init__(self):
        self.actions = ((0, 0), (0, 1), (0, 2),
                        (1, 0), (1, 1), (1, 2),
                        (2, 0), (2, 1), (2, 2))
        self.condition = (((0, 0), (0, 1), (0, 2)),
                          ((1, 0), (1, 1), (1, 2)),
                          ((2, 0), (2, 1), (2, 2)),
                          ((0, 0), (1, 0), (2, 0)),
                          ((0, 1), (1, 1), (2, 1)),
                          ((0, 2), (1, 2), (2, 2)),
                          ((0, 0), (1, 1), (2, 2)),
                          ((0, 2), (1, 1), (2, 0))
                          )
        self.field = np.zeros((3, 3), dtype=np.float32)

    @property
    def valid(self):
        return np.where(self.field == 0, 1, 0)

    @property
    def positive(self):
        return np.where(self.field == 1, 1, 0)

    @property
    def negative(self):
        return np.where(self.field == -1, 1, 0)

    def reset(self):
        self.field = np.zeros((3, 3), dtype=np.float32)

    def put(self, action, player):
        r = 0
        val = 0
        y, x = self.actions[int(action)]
        if self.field[y, x] == 0:
            self.field[y, x] = player
        else:
            r = 1
            val = 2 if player == 1 else -2
        return r, val

    def check_game_over(self, current_player):
        for cond in self.condition:
            if self.field[cond[0]] == self.field[cond[1]] == self.field[cond[2]]:
                if self.field[cond[0]] != 0:
                    return True, current_player
        if np.count_nonzero(self.field) == 9:
            return True, 0
        else:
            return False, 0

    def get_valid(self):
        valid = np.where(self.field == 0)
        res = list()
        for a1, a2 in zip(valid[0], valid[1]):
            res.append(self.actions.index((a1, a2)))
        return np.array(res)

    def sample(self):
        valid = self.get_valid()
        return np.random.choice(valid)
