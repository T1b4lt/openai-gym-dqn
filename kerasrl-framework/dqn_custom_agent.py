from rl.agents.dqn import DQNAgent
import warnings
from copy import deepcopy

import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import History

from rl.callbacks import (
    CallbackList,
    TestLogger,
    TrainEpisodeLogger,
    TrainIntervalLogger,
    Visualizer
)

import pickle
from sklearn.cluster import KMeans


class DQNCustomAgent(DQNAgent):
    def test(self, env, tuple_csv_name, reward_csv_name, defense=False, kmeans_filepath=None, max_distances=None, nb_episodes=1, action_repetition=1, callbacks=None, visualize=True,
             nb_max_episode_steps=None, nb_max_start_steps=0, start_step_policy=None, verbose=1):
        """Callback that is called before training begins.
        # Arguments
            env: (`Env` instance): Environment that the agent interacts with. See [Env](#env) for details.
            nb_episodes (integer): Number of episodes to perform.
            action_repetition (integer): Number of times the agent repeats the same action without
                observing the environment again. Setting this to a value > 1 can be useful
                if a single action only has a very small effect on the environment.
            callbacks (list of `keras.callbacks.Callback` or `rl.callbacks.Callback` instances):
                List of callbacks to apply during training. See [callbacks](/callbacks) for details.
            verbose (integer): 0 for no logging, 1 for interval logging (compare `log_interval`), 2 for episode logging
            visualize (boolean): If `True`, the environment is visualized during training. However,
                this is likely going to slow down training significantly and is thus intended to be
                a debugging instrument.
            nb_max_start_steps (integer): Number of maximum steps that the agent performs at the beginning
                of each episode using `start_step_policy`. Notice that this is an upper limit since
                the exact number of steps to be performed is sampled uniformly from [0, max_start_steps]
                at the beginning of each episode.
            start_step_policy (`lambda observation: action`): The policy
                to follow if `nb_max_start_steps` > 0. If set to `None`, a random action is performed.
            log_interval (integer): If `verbose` = 1, the number of steps that are considered to be an interval.
            nb_max_episode_steps (integer): Number of steps per episode that the agent performs before
                automatically resetting the environment. Set to `None` if each episode should run
                (potentially indefinitely) until the environment signals a terminal state.
        # Returns
            A `keras.callbacks.History` instance that recorded the entire training process.
        """
        if not self.compiled:
            raise RuntimeError(
                'Your tried to test your agent but it hasn\'t been compiled yet. Please call `compile()` before `test()`.')
        if action_repetition < 1:
            raise ValueError(
                f'action_repetition must be >= 1, is {action_repetition}')

        self.training = False
        self.step = 0

        if defense:
            kmeans_model = pickle.load(open(kmeans_filepath, "rb"))

        # Creo el dataframe en el que voy a guardar las tuplas (observacion, accion, estado_siguiente, recompensa)
        num_space_features = env.observation_space.shape[0]
        header_array = []
        for i in range(num_space_features):
            header_array.append('state_' + str(i))
        header_array.append('action')
        for i in range(num_space_features):
            header_array.append('next_state_' + str(i))
        header_array.append('reward')
        tuple_dataframe = pd.DataFrame(
            columns=header_array)

        # Creo el dataframe en el que voy a guardar las recompensas de cada epoca
        reward_dataframe = pd.DataFrame(
            columns=['episode_reward'])

        callbacks = [] if not callbacks else callbacks[:]

        if verbose >= 1:
            callbacks += [TestLogger()]
        if visualize:
            callbacks += [Visualizer()]
        history = History()
        callbacks += [history]
        callbacks = CallbackList(callbacks)
        if hasattr(callbacks, 'set_model'):
            callbacks.set_model(self)
        else:
            callbacks._set_model(self)
        callbacks._set_env(env)
        params = {
            'nb_episodes': nb_episodes,
        }
        if hasattr(callbacks, 'set_params'):
            callbacks.set_params(params)
        else:
            callbacks._set_params(params)

        self._on_test_begin()
        callbacks.on_train_begin()
        for episode in range(nb_episodes):
            callbacks.on_episode_begin(episode)
            episode_reward = 0.
            episode_step = 0

            # Obtain the initial observation by resetting the environment.
            self.reset_states()
            observation = deepcopy(env.reset())
            if self.processor is not None:
                observation = self.processor.process_observation(observation)
            assert observation is not None

            # Perform random starts at beginning of episode and do not record them into the experience.
            # This slightly changes the start position between games.
            nb_random_start_steps = 0 if nb_max_start_steps == 0 else np.random.randint(
                nb_max_start_steps)
            for _ in range(nb_random_start_steps):
                if start_step_policy is None:
                    action = env.action_space.sample()
                else:
                    action = start_step_policy(observation)
                if self.processor is not None:
                    action = self.processor.process_action(action)
                callbacks.on_action_begin(action)
                observation, r, done, info = env.step(action)

                observation = deepcopy(observation)
                if self.processor is not None:
                    observation, r, done, info = self.processor.process_step(
                        observation, r, done, info)
                callbacks.on_action_end(action)
                if done:
                    warnings.warn(
                        f'Env ended before {nb_random_start_steps} random steps could be performed at the start. You should probably lower the `nb_max_start_steps` parameter.')
                    observation = deepcopy(env.reset())
                    if self.processor is not None:
                        observation = self.processor.process_observation(
                            observation)
                    break

            # Run the episode until we're done.
            done = False
            while not done:
                callbacks.on_step_begin(episode_step)

                action = self.forward(observation)
                if self.processor is not None:
                    action = self.processor.process_action(action)
                reward = 0.
                accumulated_info = {}
                for _ in range(action_repetition):
                    callbacks.on_action_begin(action)
                    next_observation, r, d, info = env.step(action)

                    if self.processor is not None:
                        next_observation, r, d, info = self.processor.process_step(
                            next_observation, r, d, info)

                    # Creo la tupla para el dataframe y para comprobar si es anomala
                    tuple_data = []
                    for feature in observation:
                        tuple_data.append(feature)
                    tuple_data.append(action)
                    for feature in next_observation:
                        tuple_data.append(feature)
                    tuple_data.append(r)

                    # AQUI IRÁ EL BLOQUE DE DEFENSA - DEPENDIENTE DE UN PARAMETRO "DEF=True"
                    # 1 COJO LA TUPLA (OBSERVACION, ACCION, ESTADO_SIGUIENTE, RECOMPENSA)
                    # 2 MIRO SI LA TUPLA ES ANOMALA O NORMAL
                    # 3 SI ES ANOMALA, SUSTITUYO EL ESTADO_SIGUIENTE POR ALGO QUE NO SEA ANOMALO
                    if defense:
                        # Comprobamos si la tupla es anomala de alguna manera
                        distances_2_centroids = kmeans_model.transform(
                            [tuple_data])
                        closest_centroid = np.argmin(distances_2_centroids)
                        anomal_tuple = False
                        if distances_2_centroids[0][closest_centroid] > max_distances[closest_centroid]*0.8:
                            anomal_tuple = True

                        # Si la tupla es anomala, hacemos algo
                        if anomal_tuple:
                            next_observation = observation
                            # creamos de nuevo la tupla sana
                            tuple_data = []
                            for feature in observation:
                                tuple_data.append(feature)
                            tuple_data.append(action)
                            for feature in next_observation:
                                tuple_data.append(feature)
                            tuple_data.append(r)

                    # only append tuple_data if that row is not duplicated in the dataframe
                    if not (tuple_dataframe == tuple_data).all(1).any():
                        tuple_dataframe.loc[len(tuple_dataframe)] = tuple_data

                    observation = deepcopy(next_observation)

                    callbacks.on_action_end(action)
                    reward += r
                    for key, value in info.items():
                        if not np.isreal(value):
                            continue
                        if key not in accumulated_info:
                            accumulated_info[key] = np.zeros_like(value)
                        accumulated_info[key] += value
                    if d:
                        done = True
                        break
                if nb_max_episode_steps and episode_step >= nb_max_episode_steps - 1:
                    done = True
                self.backward(reward, terminal=done)
                episode_reward += reward

                step_logs = {
                    'action': action,
                    'observation': observation,
                    'reward': reward,
                    'episode': episode,
                    'info': accumulated_info,
                }
                callbacks.on_step_end(episode_step, step_logs)
                episode_step += 1
                self.step += 1

            # We are in a terminal state but the agent hasn't yet seen it. We therefore
            # perform one more forward-backward call and simply ignore the action before
            # resetting the environment. We need to pass in `terminal=False` here since
            # the *next* state, that is the state of the newly reset environment, is
            # always non-terminal by convention.
            self.forward(observation)
            self.backward(0., terminal=False)

            # Report end of episode.
            episode_logs = {
                'episode_reward': episode_reward,
                'nb_steps': episode_step,
            }
            callbacks.on_episode_end(episode, episode_logs)

            reward_dataframe.loc[len(reward_dataframe)] = [episode_reward]

        callbacks.on_train_end()
        self._on_test_end()

        # Exporto el dataframe en un csv
        tuple_dataframe.to_csv(tuple_csv_name, index=False)
        reward_dataframe.to_csv(reward_csv_name, index=False)

        return history
