import tensorflow as tf
import numpy as np
from collections import namedtuple

Batch = namedtuple("Batch", ["si", "actions", "advantages", "target_R", "terminal"])

def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer

