import numpy as np
import random

from collections import namedtuple
Batch = namedtuple("Batch", ["obses_t", "actions", "rewards", "values", "obses_tp1", "new_values", "dones"])

class ReplayBuffer(object):
    def __init__(self, size):
        """Create Prioritized Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, obs_t, action, reward, value, obs_tp1, new_value, done):
        data = (obs_t, action, reward, value, obs_tp1, new_value, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize



    def _encode_sample(self, idxes):
        obses_t, actions, rewards, values, obses_tp1, new_values, dones = [], [], [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, value, obs_tp1, new_value, done = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            values.append(value)
            new_values.append(new_value)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
        return Batch(np.array(obses_t), np.array(actions), np.array(rewards), np.array(values), \
               np.array(obses_tp1), np.array(new_values), np.array(dones))

    def sample(self, batch_size):
        """Sample a batch of experiences.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.

        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)