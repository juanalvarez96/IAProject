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

# The structure of this python code file was extracted from GitHub
# See [2] in README

# Initialize
env = gym.make('Enduro-v0')
observation = env.reset()

# Prepare the new screen for the neural network using the 
#preprocessing scheme used to create the dataset
def preprocess_screen(obs_t):
  # Source is [1] (see README)
  complete=np.dot(obs_t[..., :3], [0.2989, 0.5870, 0.1140]).flatten().reshape(1, -1).tolist()[0]
  return complete

# Load the trained model already created form the notebook
json_file = open('model.json', 'r')
loaded = json_file.read()
json_file.close()
model = model_from_json(loaded)

while True:
  #Prepare screen
  env.render()
  # Prepare the observed image
  cur_x = preprocess_screen(observation)
  # Reshape it to fit the model input
  cur_x = np.array(cur_x).reshape(1,-1)
  # Predict next action and print it
  action = model.predict_classes(cur_x)[0]
  print(action)
  #pdb.set_trace()
  # Send action to the game
  observation, _,_,_ = env.step(action)
