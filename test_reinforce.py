import gym
import matplotlib.pyplot as plt
from reinforce_keras import Agent
import numpy as np
import pdb
#from utils import plotLearning
def pong_preprocess_screen(I):
  I=np.dot(I[..., :3], [0.2989, 0.5870, 0.1140])
  #pdb.set_trace()
  return I.astype(np.float).ravel()



agent = Agent(ALPHA = 0.0005, input_dims=210*160, GAMMA = 0.99, n_actions = 9,
            layer1_size = 64, layer2_size = 64)

env = gym.make('Enduro-v0')
score_history = []

n_episodes = 2000
for i in range(n_episodes):
    done = False
    score = 0
    observation = pong_preprocess_screen(env.reset())

    while (not done):
        #pdb.set_trace()
        action = agent.choose_action(observation)
        observation_, reward, done, info = env.step(action)
        observation_=pong_preprocess_screen(observation_)
        agent.store_observation(observation, action, reward)
        observation=observation_
        score += reward
    score_history.append(score)

    cost = agent.learn()

    #pdb.set_trace()
    agent.probabilities = np.reshape(agent.probabilities, (9,-1))
    #plt.plot(agent.probabilities.sum(axis = 1))
    #plt.show()
    agent.probabilities=np.zeros(9)
    print('Episode {}, score {}, average_score {}, cost {}'.format(i, score, np.mean(score_history[-100:]), int(cost)))

        