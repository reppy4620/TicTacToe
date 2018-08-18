import chainer.optimizer as optim
from chainer.optimizers import Adam

from chainerrl.agents import DoubleDQN

from chainerrl.replay_buffer import PrioritizedEpisodicReplayBuffer
from chainerrl.explorers import LinearDecayEpsilonGreedy

from .QFunction import QFunction
from .RandomAgent import RandomAgent


def get_ddqn(env):
    rda = RandomAgent(env)
    q_func = QFunction()
    opt = Adam(alpha=1e-3)
    opt.setup(q_func)
    opt.add_hook(optim.GradientClipping(1.0), 'hook')

    rbuf = PrioritizedEpisodicReplayBuffer(5 * 10 ** 5)
    explorer = LinearDecayEpsilonGreedy(start_epsilon=1.0, end_epsilon=0.3, decay_steps=10000,
                                        random_action_func=rda.random_action)

    agent = DoubleDQN(q_func, opt, rbuf, gamma=0.995, explorer=explorer, replay_start_size=500,
                      target_update_interval=1, target_update_method='soft',
                      update_interval=4, soft_update_tau=1e-2, n_times_update=1,
                      gpu=0, minibatch_size=128
                      )
    return agent
