from Agents import get_ddqn
from Game import Environment

import random as rd


class Result:

    def __init__(self):
        self.win = 0
        self.lose = 0
        self.draw = 0

    def print(self):
        print(f'Win: {self.win}, Lose: {self.lose}, Draw: {self.draw}')
        self.reset()

    def reset(self):
        self.win = 0
        self.lose = 0
        self.draw = 0


class Arena:

    def __init__(self):
        self.p1 = Result()
        self.p2 = Result()
        self.env = Environment()

    def battle(self, agent, mode):
        game_over = False
        value = None
        res = ''
        agents = list()
        if mode == 1:
            a2 = get_ddqn(self.env)
            a2.load('models/agent2')
            agents = [None, agent, a2]
        elif mode == -1:
            a1 = get_ddqn(self.env)
            a1.load('models/agent1')
            agents = [None, a1, agent]
        assert len(agents) > 0
        obs = self.env.reset(rd.choice([-1, 1]))
        while not game_over:
            action = agents[self.env.current_player].act(obs)
            obs, _, game_over, value = self.env.step(action)
        assert value is not None
        self.env.print_board()
        if value == 1:
            self.p1.win += 1
            self.p2.lose += 1
            res = 'New Model Win'
        elif value == -1:
            self.p1.lose += 1
            self.p2.win += 1
            res = 'Current Model Win'
        elif value == 0:
            self.p1.draw += 1
            self.p2.draw += 1
            res = 'Draw'
        elif value == 2:
            self.p1.lose += 1
            self.p2.win += 1
            res = 'P1 missed'
        elif value == -2:
            self.p1.win += 1
            self.p2.lose += 1
            res = 'P2 missed'

        return res

    def battles(self, a1, a2):
        self.p1.reset()
        self.p2.reset()
        print('-' * 20)
        print('Arena')
        for i in range(1, 11):
            res = self.battle(a1, 1)
            print(f'Battle: {i}')
            print(res)
        res1 = 1 if self.p1.win > self.p2.win else 0
        self.p1.reset()
        self.p2.reset()
        for i in range(1, 11):
            res = self.battle(a2, -1)
            print(f'Battle: {i}')
            print(res)
        res2 = 1 if self.p1.win < self.p2.win else 0
        print('-' * 20)
        return res1, res2
