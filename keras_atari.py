import os
import gym
import cv2
import argparse
import sys, glob
import numpy as np
from keras import backend as K
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, Dense, Reshape
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import Adam, Adamax, RMSprop
from keras.layers.advanced_activations import PReLU
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation, Dropout, Flatten
from keras.layers.convolutional import UpSampling2D, Convolution2D
from keras.models import model_from_json # Needed to load out neural network
import pdb
import imageio

# Initialize
env = gym.make('Enduro-v0')
number_of_inputs = env.action_space.n #This is incorrect for Pong (?)
observation = env.reset()

def preprocess_screen(obs_t):
    complete=np.dot(obs_t[..., :3], [0.2989, 0.5870, 0.1140]).flatten().reshape(1, -1).tolist()[0]
    return complete

json_file = open('model.json', 'r')
loaded = json_file.read()
json_file.close()
model = model_from_json(loaded)

while True:
  #if render: 
  env.render()
  #Preprocess, consider the frame difference as features
  # pdb.set_trace()
  cur_x = preprocess_screen(observation)
  cur_x = np.array(cur_x).reshape(1,-1)
  action = model.predict_classes(cur_x)[0]
  print(action)
  #pdb.set_trace()
  #print(value)
  observation, _,_,_ = env.step(action)
