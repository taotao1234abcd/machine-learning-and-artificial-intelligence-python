import unittest
from functools import partial

import gym
import tensorflow as tf

from spinup import sac


env_fn = partial(gym.make, 'Pendulum-v0')
with tf.Graph().as_default():
    sac(env_fn)
