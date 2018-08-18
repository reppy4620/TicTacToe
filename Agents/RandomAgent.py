class RandomAgent:

    def __init__(self, env):
        self.env = env

    def random_action(self):
        return self.env.sample()
