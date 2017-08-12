
import tensorflow as tf
import numpy as np
from auditor import Auditor
from time import time
import logging
import brain
from replay_buffer import ReplayBuffer

class WorkerFactory(object):
    def create_worker(**kwargs):
        algo_name = kwargs['algo_name']

        if (algo_name == 'dqn'):
            return WorkerDQN(**kwargs)


class GeneralWorker(object):
    def __init__(self, **kwargs):
        self.env = kwargs['env']
        self.name = kwargs['worker_name']
        self.initial_learning_rate = kwargs['learning_rate']
        self.algo_name = kwargs['algo_name']

        self.max_master_time_step = kwargs['max_master_time_step']
        self.max_clock_limit = kwargs['max_clock_limit']  

        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_mean_values = []
        self.local_timesteps = 0
        self.auditor = Auditor(**kwargs)

        self.anneal_learning_rate = kwargs['anneal_learning_rate']
        self.start_clock = time()
        self.use_clock = kwargs['anneal_by_clock']
        self.max_clock_limit = kwargs['max_clock_limit']

        self.log = logging.getLogger(__name__)
        self.log.setLevel(logging.WARN)

        self.master_timestep = tf.get_variable("master_timestep", [], tf.int32,
                                               initializer=tf.constant_initializer(0, dtype=tf.int32),
                                               trainable=False)

        self.rollout_size = tf.placeholder(shape=[], dtype=tf.int32, name='rollout_size')
        self.inc_step = self.master_timestep.assign_add(self.rollout_size, use_locking=True)



class WorkerDQN(GeneralWorker):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.local_brain = brain.DQN(
            **kwargs
        )
        self.replay_buffer = ReplayBuffer(kwargs['replay_buffer_size'])
        self.beta_schedule = None
        self.replay_start_size = kwargs['replay_start_size']
        self.train_update_freq = kwargs['train_update_freq']
        self.minibatch_size = kwargs['minibatch_size']
        self.target_network_update_freq = kwargs['target_network_update_freq']


    def work(self, session):
        print("Starting " + self.name)

        while True:
            with tf.variable_scope(self.name):
                with session.as_default(), session.graph.as_default():
                    last_state = self.env.reset()
                    for t in range(self.max_master_time_step):

                        new_state, terminal = self._env_sampling(session, last_state, t)
                        last_state = new_state

                        if terminal:
                            last_state = self.env.reset()

                        if t >  self.replay_start_size and t % self.train_update_freq == 0:
                            replay_batch = self.replay_buffer.sample(self.minibatch_size)
                            feed_dict = {
                                self.local_brain.learning_rate: self.initial_learning_rate,
                                self.local_brain.input: replay_batch.obses_t,
                                self.local_brain.input_tp1: replay_batch.obses_tp1,
                                self.local_brain.r: replay_batch.rewards,
                                self.local_brain.gamma: 0.99,
                                self.local_brain.actions: np.vstack(replay_batch.actions)
                            }
                            session.run(self.local_brain.update_gradient_op, feed_dict = feed_dict)

                        if t > self.replay_start_size and t % self.target_network_update_freq == 0:
                            replay_batch = self.replay_buffer.sample(self.minibatch_size)
                            feed_dict = {
                                self.local_brain.learning_rate: self.initial_learning_rate,
                                self.local_brain.input: replay_batch.obses_t,
                                self.local_brain.input_tp1: replay_batch.obses_tp1,
                                self.local_brain.r: replay_batch.rewards,
                                self.local_brain.gamma: 0.99,
                                self.local_brain.actions: np.vstack(replay_batch.actions)
                            }
                            session.run(self.local_brain.update_target_ops, feed_dict=feed_dict)

                        if t >= 10000 and t % 5000 == 0:
                            feed_dict = {
                                self.local_brain.input: np.asarray(self.auditor.random_states)
                            }
                            value = session.run(self.local_brain.max_q_eval, feed_dict=feed_dict)
                            self.auditor.recordMaxQEval(value, t)

    def _env_sampling(self, session, last_state, t):

        self.auditor.collectRandomState(last_state)

        fetched = self.local_brain.get_transition(session, last_state, t)
        action, value_ = fetched[0], fetched[1]

        new_state, reward, terminal, info = self.env.step(action.argmax())

        new_state_value_ = 0
        if not terminal:
            fetched = self.local_brain.get_transition(session, new_state, t)
            new_state_value_ = fetched[1]

        self.auditor.appendReward(reward, terminal)

        self.replay_buffer.add(last_state, action, reward,
                               value_, new_state, new_state_value_, float(terminal))   
        return new_state, terminal
