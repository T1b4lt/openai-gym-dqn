from __future__ import division
import argparse

from PIL import Image
import numpy as np
import gym

import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Convolution2D, Permute
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint
from dqn_custom_agent import DQNCustomAgent


class MountaincarProcessor(Processor):
    def process_observation(self, observation):
        perturbation = 0
        randnum = random.random()
        if randnum < 0.5:
            perturbation = 0.05
        return observation + perturbation


parser = argparse.ArgumentParser()
parser.add_argument(
    '--mode', choices=['train', 'testa', 'testd'], default='train')
parser.add_argument('--env-name', type=str, default='MountainCar-v0')
parser.add_argument('--weights', type=str, default=None)
args = parser.parse_args()

# Get the environment and extract the number of actions.
env = gym.make(args.env_name)
env._max_episode_steps = 1000
# np.random.seed(123)
# env.seed(123)
nb_actions = env.action_space.n

# Next, we build a very simple model.
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())

# Finally, we configure and compile our agent. You can use every built-in tensorflow.keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=100000, window_length=1)
processor = MountaincarProcessor()

# Select a policy. We use eps-greedy action selection, which means that a random action is selected
# with probability eps. We anneal eps from 1.0 to 0.1 over the course of 100K steps. This is done so that
# the agent initially explores the environment (high eps) and then gradually sticks to what it knows
# (low eps). We also set a dedicated eps value that is used during testing. Note that we set it to 0.05
# so that the agent still performs some random actions. This ensures that the agent cannot get stuck.
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05,
                              nb_steps=100000)

if args.mode == 'train':
    # The trade-off between exploration and exploitation is difficult and an on-going research topic.
    # If you want, you can experiment with the parameters or use a different policy. Another popular one
    # is Boltzmann-style exploration:
    # policy = BoltzmannQPolicy(tau=1.)
    # Feel free to give it a try!
    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=1000,
                   target_model_update=1e-2, policy=policy, gamma=.99, processor=processor)
    dqn.compile(Adam(learning_rate=1e-3), metrics=['mae'])

    # Okay, now it's time to learn something! We capture the interrupt exception so that training
    # can be prematurely aborted. Notice that now you can use the built-in tensorflow.keras callbacks!
    weights_filename = f'checkpoints/mountaincar/dqn_{args.env_name}_weights.h5f'
    checkpoint_weights_filename = 'checkpoints/mountaincar/dqn_' + \
        args.env_name + '_weights_{step}.h5f'
    log_filename = f'checkpoints/mountaincar/dqn_{args.env_name}_log.json'
    callbacks = [ModelIntervalCheckpoint(
        checkpoint_weights_filename, interval=20000)]
    callbacks += [FileLogger(log_filename, interval=5000)]
    history = dqn.fit(env, callbacks=callbacks,
                      nb_steps=200000, log_interval=5000)

    # After training is done, we save the final weights one more time.
    dqn.save_weights(weights_filename, overwrite=True)

    # Finally, evaluate our algorithm for 10 episodes.
    dqn.test(env, nb_episodes=5, visualize=True)
elif args.mode == 'testa':
    dqn = DQNCustomAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=1000,
                         target_model_update=1e-2, policy=policy, gamma=.99, processor=processor)
    dqn.compile(Adam(learning_rate=1e-3), metrics=['mae'])

    weights_filename = f'checkpoints/mountaincar/dqn_{args.env_name}_weights.h5f'
    if args.weights:
        weights_filename = args.weights
    dqn.load_weights(weights_filename)
    dqn.test(env, tuple_csv_name="ejecuciones_mountaincar/mountaincar_attack.csv", reward_csv_name="ejecuciones_mountaincar/mountaincar_reward_attack.csv",
             nb_episodes=1000, visualize=False)

elif args.mode == 'testd':
    dqn = DQNCustomAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=1000,
                         target_model_update=1e-2, policy=policy, gamma=.99, processor=processor)
    dqn.compile(Adam(learning_rate=1e-3), metrics=['mae'])

    weights_filename = f'checkpoints/mountaincar/dqn_{args.env_name}_weights.h5f'
    if args.weights:
        weights_filename = args.weights
    dqn.load_weights(weights_filename)
    dqn.test(env,
             tuple_csv_name="ejecuciones_mountaincar/mountaincar_defense.csv",
             reward_csv_name="ejecuciones_mountaincar/mountaincar_reward_defense.csv",
             defense=True,
             classification_csv_name="resultados_clasificadores/mountaincar_classification.csv",
             anomaly_method=2,
             substitution_method=1,
             kmeans_filepath="notebooks/kmeans_mountaincar_2048c_norm.pkl",
             tuples_filepath="ejecuciones_mountaincar/mountaincar_noattack.csv",
             threshold=50,
             nb_episodes=5,
             visualize=False)
