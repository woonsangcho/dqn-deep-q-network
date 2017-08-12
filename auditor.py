
import tensorflow as tf
import numpy as np
import os

"""
For evaluating agents
"""
class Auditor():
    def __init__(self, **kwargs):
        self.name = "auditor_for_" + kwargs['worker_name']

        self.episode_rewards = [0]
        self.episode_length_counter = 0
        self.episode_lengths = [0]
        self.local_timesteps = 0
        self.episode_counter = 1

        logdirpath = str(os.getcwd())+'/' \
                     + kwargs['env_id'] + '_' \
                     + kwargs['algo_name'] + '_' \
                     + str(kwargs['learning_rate']) + '/' \
                     + self.name

        self.summary_writer = tf.summary.FileWriter(logdir = logdirpath)

        self.random_states = []

    def appendReward(self, reward, terminal=False, past_ep_size = 3):
        self.episode_rewards[-1] += reward
        self.episode_lengths[-1] += 1

        if terminal is True:
            if len(self.episode_rewards) == past_ep_size:
                self.recordSummary(past_ep_size)
                self.episode_rewards.pop(0)
                self.episode_lengths.pop(0)

            self.episode_rewards.append(0)
            self.episode_lengths.append(0)
            self.episode_counter += 1


    def recordSummary(self, past_ep_size = 3):

        mean_reward = np.round(np.mean(self.episode_rewards[-min(past_ep_size, len(self.episode_rewards)):]), 1)
        mean_length = np.round(np.mean(self.episode_lengths[-min(past_ep_size, len(self.episode_lengths)):]), 1)

        summary = tf.Summary()
        summary.value.add(tag='Stats/MeanReward', simple_value=int(mean_reward))
        summary.value.add(tag='Stats/MeanLength', simple_value=int(mean_length))
        self.summary_writer.add_summary(summary, self.episode_counter)
        self.summary_writer.flush()

    def collectRandomState(self, state):
        if len(self.random_states) < 10000:
            self.random_states += [state]

    def recordMaxQEval(self, max_q_eval, t):
        summary = tf.Summary()
        summary.value.add(tag='Stats/max_q_eval', simple_value=max_q_eval)
        self.summary_writer.add_summary(summary, t)
        self.summary_writer.flush()

