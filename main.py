#!/usr/bin/env bash

import tensorflow as tf
from worker import WorkerFactory
from environment import create_env
from argparser import ArgParser
import random

if __name__ == '__main__':

    parser = ArgParser()
    args = parser.parse_args()

    if (args.anneal_by_clock):
        assert isinstance(args.max_clock_limit, float)
        assert args.max_clock_limit > 0

    tf.reset_default_graph()

    learning_rate = random.uniform(0.0001,0.0005)
    
    env = create_env(args.env_id)
    vars(args)['env'] = env
    vars(args)['learning_rate'] = learning_rate

    local_env = create_env(args.env_id)
    local_name = "worker_0"

    vars(args)['env'] = local_env
    vars(args)['worker_name'] = local_name

    worker = WorkerFactory.create_worker(**vars(args))

    config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=True)

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        worker.work(sess)
