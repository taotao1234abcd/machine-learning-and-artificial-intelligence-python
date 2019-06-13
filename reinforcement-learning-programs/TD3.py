import unittest
from functools import partial

import gym
import tensorflow as tf

from spinup import td3


env_fn = partial(gym.make, 'Pendulum-v0')
with tf.Graph().as_default():
    td3(env_fn)
