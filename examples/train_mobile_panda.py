import gymnasium as gym
from stable_baselines3 import DDPG, HerReplayBuffer

import panda_gym
import imageio

env = gym.make("PandaPickAndPlace-v3")

model = DDPG(policy="MultiInputPolicy", env=env, replay_buffer_class=HerReplayBuffer, verbose=1)

model.learn(total_timesteps=1000)


images = []

observation, info = env.reset()
images.append(env.render())

for _ in range(1000):
    action, _states = model.predict(observation, deterministic=False)
    observation, reward, terminated, truncated, info = env.step(action)
    images.append(env.render())

    if terminated or truncated:
        observation, info = env.reset()
        images.append(env.render())

env.close()


imageio.mimsave('movie.gif', images)
