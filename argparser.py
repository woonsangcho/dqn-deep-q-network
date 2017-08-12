
import argparse
import os

class ArgParser():
    def __init__(self):
        self.parser = argparse.ArgumentParser(description=None)
        self._add_arguments()

    def parse_args(self):
        return self.parser.parse_args()

    def _add_arguments(self):

        self.parser.add_argument('--algo-name', default="dqn", help='Name of algorithm. For list, see README')
        self.parser.add_argument('--num-workers', default=1, type=int, help='Number of workers')
        self.parser.add_argument('--log-dir', default=os.getcwd() + "/tmp", help='Log directory path')
        self.parser.add_argument('--env-id', default="PongNoFrameskip-v4", help='Environment id')  # PongNoFrameskip-v4
        self.parser.add_argument('--max-bootstrap-length', default=20, type=int, help='Max length of trajectory \
                                                                                 before bootstrapping')   #20
        self.parser.add_argument('--max-master-time-step', default=999999999999999, type=int,
                            help='Max number of time steps to train')
        self.parser.add_argument('--max-clock-limit', default=0, type=float, help='Max clock limit to train')
        self.parser.add_argument('--anneal-learning-rate', action='store_true',
                            help='Flag to whether to anneal learning rate or not')
        self.parser.add_argument('--anneal-by-clock', action='store_true', help='Flag to anneal learning rate by clock time')
        self.parser.add_argument('--use-gpu', action='store_true', help='Flag to use gpu')

        def conv_layer_type(inpt):
            try:
                print(inpt)
                tup = eval(inpt)
                return tup
            except:
                raise argparse.ArgumentTypeError("Type in a list of 3-valued tuples e.g. [(16, 8, 4), (32, 4, 2)]\
                                                 where first value: # of filters, second value: 1-dim size of squared filter, \
                                                 third value: stride value")

        self.parser.add_argument('--convs', nargs='*', default=[(16, 8, 4), (32, 4, 2)],
                            help="Convolutional layer specification", type=conv_layer_type)
        self.parser.add_argument('--hiddens', nargs='*', type=int, default=[256],
                            help="Hidden layer specification: Type in a list of integers e.g. [256 256] where each element\
                                                 denotes the hidden layer node sizes in order given")
        self.parser.add_argument('--replay-buffer-size', default=1000000, type=int, help='Replay memory size')
        self.parser.add_argument('--exploration-fraction', default=0.1, type=float,
                            help='Exploration fraction, after which final eps is used')
        self.parser.add_argument('--exploration-final-eps', default=0.05, type=float,
                            help='Exploration afinal eps after exploration fraction * max time step.')
        self.parser.add_argument('--replay-start-size', default=50000, type=int,
                            help='random policy timesteps before actual learning begins')
        self.parser.add_argument('--train-update-freq', default=5, type=int,
                            help='number of actions between successive SGD updates')  #4
        self.parser.add_argument('--minibatch-size', default=32, type=int, help='minibatch size for SGD')
        self.parser.add_argument('--target-network-update-freq', default=10000, type=int,
                            help='target network update freq to stabilize learning')

