from keras.layers import Dense, Activation, Input
from keras.models import Model, load_model
from keras.optimizers import Adam
import keras.backend as K
import numpy as np
import pdb


class Agent(object):
  def __init__(self, ALPHA, GAMMA = 0.99, n_actions = 9, layer1_size = 16,
                 layer2_size = 16, input_dims = 210*160, fname = 'reinforce.h5'):
    self.gamma = GAMMA
    self.lr = ALPHA
    self.G = 0
    self.input_dims = input_dims
    self.fc1_dims = layer1_size
    self.fc2_dims = layer2_size
    self.n_actions = n_actions
    self.state_memory = []
    self.action_memory = []
    self.reward_memory = []
    self.probabilities = np.zeros(9)

    self.policy, self.predict = self.build_policy_network()
    self.action_space = [ i for i in range(n_actions)]
    self.mode_file = fname

  def build_policy_network(self):
    input = Input(shape=(self.input_dims,))
    advantages = Input(shape=[1])
    dense1 = Dense(self.fc1_dims, activation = 'relu')(input)
    dense2 = Dense(self.fc2_dims, activation = 'relu')(dense1)
    probs = Dense(self.n_actions, activation='softmax')(dense2)

    def custom_loss(y_true, y_pred):
      out = K.clip(y_pred, 1e-8, 1-1e-8) # To avoid deividing by 0
      log_lik = y_true*K.log(out)

      return K.sum(-log_lik*advantages)

    policy = Model(input=[input, advantages], output = [probs])
    policy.compile(optimizer = Adam(lr = self.lr), loss = custom_loss)

    predict = Model(input=[input], output = [probs])

    return policy, predict

  
  def choose_action (self, observation):
    #pdb.set_trace()
    state = observation[np.newaxis, :]
    probabilities = self.predict.predict(state)[0]
    self.probabilities = np.concatenate((probabilities, self.probabilities), axis = 0)
    action = np.random.choice(self.action_space, p = probabilities)
    return action

  def store_observation(self, observation, action, reward):
    self.action_memory.append(action)
    self.state_memory.append(observation)
    self.reward_memory.append(reward)

  def learn(self):
    #pdb.set_trace()
    state_memory = np.array(self.state_memory)
    action_memory = np.array(self.action_memory)
    reward_memory = np.array(self.reward_memory)

    actions = np.zeros([len(action_memory), self.n_actions])
    actions[np.arange(len(action_memory)), action_memory] = 1

    G = np.zeros_like(reward_memory)
    for t in range(len(reward_memory)):
      G_sum = 0
      discount = 1
      for k in range (t, len(reward_memory)):
        G_sum += reward_memory[k]*discount
        discount *=self.gamma

      G[t] = G_sum
    mean = np.mean(G)
    std = np.std(G) if np.std(G)>0 else 1
    self.G = (G-mean)/std

    cost = self.policy.train_on_batch([state_memory, self.G], actions) # y_pred y_true

    self.state_memory=[]
    self.action_memory = []
    self.reward_memory = []

    return cost

  def save_model(self):
    self.policy.save(self.model_file)

  def load_model(self):
    self.policy = load_model(self.model_file)

