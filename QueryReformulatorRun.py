import os
import time
from QueryReformulatorEnv import QueryReformulatorEnv
import argparse
import numpy as np
np.random.seed(1337)  # for reproducibility
import distutils.util
from tensorforce.agents import PPOAgent, DQNAgent, VPGAgent
from tensorforce.execution import Runner

DATA_DIR = "."
env = QueryReformulatorEnv(os.path.join(DATA_DIR, '/home/ss49/QR/'),dset='train',is_train=True,verbose=True)


# Network as list of layers
network_spec = [
    #dict(type='embedding', indices=6000, size=300),
    dict(type='flatten'),
    dict(type='dense', size=200, activation='relu'),
    dict(type='dense', size=200, activation='relu')
]

# Create a Proximal Policy Optimization agent
agent = PPOAgent(
    states_spec=env.states,
    actions_spec=env.actions,
    network_spec=network_spec,
    batch_size=64,
    step_optimizer=dict(
        type='adam',
        learning_rate=1e-4
    )
)

"""agent = DQNAgent(
    states_spec=dict(type='int', shape=(1,args.maxlen)),
    actions_spec=dict(type='int', num_actions=65),
    network_spec=network_spec,
    batch_size=1,
    first_update=1,
    target_sync_frequency=5
)"""

"""agent = VPGAgent(
    states_spec=dict(type='int', shape=(1,args.maxlen)),
    actions_spec=dict(type='int', num_actions=65),
    network_spec=network_spec,
    batch_size=1,
    optimizer=dict(
        type='adam',
        learning_rate=1e-4
    )
)"""


# Create the runner
runner = Runner(agent=agent, environment=env)


# Callback function printing episode statistics
def episode_finished(r):
    print("Finished episode {ep} after {ts} timesteps (reward: {reward})".format(ep=r.episode, ts=r.episode_timestep,
                                                                                 reward=r.episode_rewards[-1]))
    return True


# Start learning
runner.run(episodes=10, max_episode_timesteps=10, episode_finished=episode_finished)

# Print statistics
print("Learning finished. Total episodes: {ep}. Average reward of last 100 episodes: {ar}.".format(
    ep=runner.episode,
    ar=np.mean(runner.episode_rewards[-100:]))
)
