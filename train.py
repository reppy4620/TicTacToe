from Game import Environment
from Agents import get_ddqn
from arena import Arena

import argparse


def train(n_epoch):
    env = Environment()
    arena = Arena()
    print('Making Models...')
    agents = [None, get_ddqn(env), get_ddqn(env)]

    for e in range(n_epoch):
        print('-' * 30)
        print(f'{e} Epoch Start...')
        print('-' * 30)
        for i in range(100):
            obs = env.reset(1)
            done = False
            rewards = [0] * 4
            value = 0
            last = None
            while not done:
                last = env.state
                action = agents[env.current_player].act_and_train(obs, rewards[env.current_player])
                obs, r, done, value = env.step(action)
                rewards[env.current_player-1] = r
            assert last is not None
            if value == 1:
                agents[1].stop_episode_and_train(env.state, 1, True)
                agents[-1].stop_episode_and_train(last, -1, True)
            elif value == -1:
                agents[-1].stop_episode_and_train(env.state, 1, True)
                agents[1].stop_episode_and_train(last, -1, True)
            elif value == 0:
                agents[1].stop_episode_and_train(env.state, 0, True)
                agents[-1].stop_episode_and_train(env.state, 0, True)
            elif value == 2:
                agents[1].stop_episode_and_train(env.state, -1, True)
                agents[-1].stop_episode()
            elif value == -2:
                agents[-1].stop_episode_and_train(env.state, -1, True)
                agents[1].stop_episode()
            # print(env.print_board())
            # print('Result: ', value)
            print(f'{i} Episode Ended')
        res1, res2 = arena.battles(agents[1], agents[2])
        if res1:
            dir_name = 'models/' + 'agent1'
            print('Saving model', dir_name)
            agents[1].save(dir_name)
            agents[1].replay_buffer.save('models/replay1.npz')
        if res2:
            dir_name = 'models/' + 'agent2'
            print('Saving model', dir_name)
            agents[2].save(dir_name)
            agents[2].replay_buffer.save('models/replay2.npz')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('n_epoch', type=int, default=10000)
    args = parser.parse_args()
    train(args.n_epoch)
