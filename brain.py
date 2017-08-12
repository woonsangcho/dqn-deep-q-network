
import tensorflow as tf
import models
import tensorflow.contrib.layers as layers
from mathlib import LinearSchedule
from mathlib import huber_loss

class Brain(object):
    def __init__(self, **kwargs):
        self.env = kwargs['env']
        self.scope = kwargs['worker_name']

    def get_transition(self, session, inpt,t):
        raise NotImplementedError

    def get_value(self, session, inpt):
        raise NotImplementedError

class NeuralNetwork(Brain):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        with tf.variable_scope(self.scope):
            self.input = tf.placeholder(tf.float32,
                                        [None] + list(self.env.observation_space.shape)
                                        , name='input')

            self.cnn_output = models.CNN(scope='cnn',
                                    convs=kwargs['convs'],
                                    hiddens=kwargs['hiddens'],
                                    inpt=self.input)

            self.mlp_output = models.MLP(scope='mlp',
                                    hiddens=kwargs['hiddens'],
                                    inpt=self.cnn_output)


class DQN(NeuralNetwork):
    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        with tf.variable_scope(self.scope):
            with tf.variable_scope('qval'):
                self.qval_t = layers.fully_connected(self.mlp_output, num_outputs=self.env.action_space.n,
                                            activation_fn=None
                                            )

        self.max_q_over_actions = tf.reduce_max(self.qval_t, axis=1)
        self.max_q_eval = tf.reduce_max(tf.reduce_max(self.qval_t, 1),)

        exploration_epsilon = LinearSchedule(schedule_timesteps=1000000,
                                             # set custom -- int(kwargs['exploration_fraction'] * kwargs['max_master_time_step'])
                                             initial_p=1.0,
                                             final_p=kwargs['exploration_final_eps'])

        self.timestep = tf.placeholder(shape=[], dtype=tf.int32, name='timestep')

        choose_random_bool = tf.random_uniform([1], minval=0, maxval=1, dtype=tf.float64) \
                       < exploration_epsilon.value(self.timestep)  # bool
        random_action = tf.one_hot(tf.random_uniform([1], minval=0, maxval=self.env.action_space.n, dtype=tf.int64), self.env.action_space.n)
        argmax_action = tf.one_hot(tf.argmax(self.qval_t, axis=1), self.env.action_space.n)
        self.sample_action = tf.where(choose_random_bool, random_action, argmax_action)


        self.learning_rate = tf.placeholder(shape=[], dtype=tf.float32, name='learning_rate')
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate, use_locking=False)

        self.r = tf.placeholder(shape=[None], dtype=tf.float32, name='rewards')
        self.gamma = tf.placeholder(shape=[], dtype=tf.float32, name='gamma')
        self.actions = tf.placeholder(tf.float32, [None, self.env.action_space.n], name="actions")

        with tf.variable_scope('master', reuse=None):
            self.input_tp1 = tf.placeholder(tf.float32,
                                                  [None] + list(self.env.observation_space.shape)
                                                  , name='input_tp1')

            cnn_output = models.CNN(scope='cnn',
                                    convs=kwargs['convs'],
                                    hiddens=kwargs['hiddens'],
                                    inpt=self.input_tp1,
                                    reuse=None)

            mlp_output = models.MLP(scope='mlp',
                                    hiddens=kwargs['hiddens'],
                                    inpt=cnn_output,
                                    reuse=None)

            with tf.variable_scope('qval', reuse=None):
                self.qval_tp1 = layers.fully_connected(mlp_output, num_outputs=self.env.action_space.n,
                                              activation_fn=None,
                                              )

        self.target_R = self.r + self.gamma * tf.reduce_max(self.qval_tp1, axis=1)

        self.value = tf.reduce_sum(self.qval_t * self.actions, 1)
        td_errors = self.value - tf.stop_gradient(self.target_R)
        self.loss = tf.reduce_sum(huber_loss(td_errors))

        update_vars_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
        self.update_gradient_op = self.optimizer.minimize(self.loss, var_list=update_vars_list)

        self.update_target_ops = []
        target_vars_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'master')
        for var, var_target in zip(sorted(update_vars_list, key=lambda v: v.name),
                                   sorted(target_vars_list, key=lambda v: v.name)):
            self.update_target_ops.append(var_target.assign(var))
        self.update_target_ops = tf.group(*self.update_target_ops)


    def get_transition(self, session, inpt, t):
        fetched = session.run([self.sample_action, self.max_q_eval],
                              {self.input: [inpt],
                               self.timestep: t})
        return fetched

