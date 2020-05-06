import os
import gym
import cv2
import argparse
import sys, glob
import numpy as np
import pandas as pd
import pdb
import keras
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

IMAGE_LENGTH = int(33600)

def load_data():
    #pdb.set_trace()
    df = pd.read_csv('dataset_v2.txt', sep = '.', header = None)
    letters = df[(df.index % (IMAGE_LENGTH+1)==0)].values.tolist()
    images = df[(df.index % (IMAGE_LENGTH+1)!=0)].values.tolist()
    n =IMAGE_LENGTH
    final = [images[i * n:(i + 1) * n] for i in range((len(images) + n - 1) // n )]  
    #pdb.set_trace()
    return letters, final

letter, final = load_data()
# Format loaded data
for i in range (0, len(final)):
    final[i] = np.concatenate(final[i])
#pdb.set_trace()
letter = np.concatenate(letter)
# Split dataset using a rule of 0.7
train_ratio = 0.7
n_train_samples = int(len(final) * train_ratio)
x_train, y_train = final[:n_train_samples], letter[:n_train_samples]
x_val, y_val = final[n_train_samples:], letter[n_train_samples:]
pdb.set_trace()

