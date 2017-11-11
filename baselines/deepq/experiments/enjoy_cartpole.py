import gym
import time

from baselines import deepq


def main():
    env = gym.make("CartPole-v0")
    act = deepq.load("cartpole_model.pkl")

    while True:
        obs, done = env.reset(), False
        episode_rew = 0
        while not done:
            obs, rew, done, _ = env.step(act(obs[None])[0])
            print(obs)
            time.sleep(0.5)
            episode_rew += rew
        print("Episode reward", episode_rew)


if __name__ == '__main__':
    main()
