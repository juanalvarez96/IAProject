import gym
from gym.utils.play import play
import imageio
import pdb
import numpy as np
import pandas as pd
import pdb
rews=[]
infos=[]
dones = []
actions=[]
IMAGE_LENGTH = int(33600)
# We are not storing the action (for the moment)
def saveToDataSetTextFile(letter, img):
    #pdb.set_trace()
    complete=np.dot(img[..., :3], [0.2989, 0.5870, 0.1140]).flatten().reshape(1, -1).tolist()[0]
    complete.insert(0,letter)
    with open('dataset.txt', "a") as myfile:
      np.savetxt(myfile, delimiter='.', X = complete, fmt = '%d')




def mycallbacks(obs_t, obs_tp1, action, rew, done, info):
    #pdb.set_trace()
    rews.append(rew)
    infos.append(info)
    dones.append(done)
    actions.append(action)
    saveToDataSetTextFile(action, obs_t)



def load_data(path):
    #pdb.set_trace()
    df = pd.read_csv('dataset.txt', sep = '.', header = None)
    letters = df[(df.index % (IMAGE_LENGTH+1)==0)].values.tolist()
    images = df[(df.index % (IMAGE_LENGTH+1)!=0)].values.tolist()
    n =IMAGE_LENGTH
    final = [images[i * n:(i + 1) * n] for i in range((len(images) + n - 1) // n )]  
    #pdb.set_trace()
    return letters, final




env = gym.make('Enduro-v0')
env.reset()

play(env, zoom=3, fps=40, callback = mycallbacks)
#pdb.set_trace()
#a, b = load_data('dataset.txt')
env.close()