from rete import ReteNeurale
import tensorflow as tf
from keras import Optimizer 
import tensorflow_probability as tfp
import gym
import numpy as np
from gym.utils import play

env = gym.make('coinrun-v1',render_mode="rgb_array")
play.play(env, zoom=3)
