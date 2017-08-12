import math
import tensorflow as tf


def log_uniform(lo, hi, rate):
  log_lo = math.log(lo)
  log_hi = math.log(hi)
  v = log_lo * (1-rate) + log_hi * rate
  return math.exp(v)

class LinearSchedule(object):
    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):

        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_p
        self.initial_p = initial_p

    def value(self, t):
        fraction = tf.minimum(tf.divide(t, self.schedule_timesteps), tf.ones([1], dtype=tf.float64))
        return fraction * (self.final_p - self.initial_p)

def huber_loss(x, delta=1.0):
    """Reference: https://en.wikipedia.org/wiki/Huber_loss"""
    return tf.where(
      tf.abs(x) < delta,  # condition
      tf.square(x) * 0.5,  # if satisfied use this
      delta * (tf.abs(x) - 0.5 * delta)  # else use this
    )

