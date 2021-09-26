import random
import torch
import numpy as np
import os
import json
class DRL(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.inception = torch.hub.load("pytorch/vision", 'inception_v3', pretrained=True, transform_input=True)
    self.dense = torch.nn.Linear(1000, 32)
    self.output = torch.nn.Linear(32, 6)
  def forward(self, env_state, car_state=None):
    temp = torch.zeros(3, 299, 299).cuda()
    input = torch.stack([temp, env_state], dim = 0).cuda()
    output = self.inception(input)
    output = self.dense(output.logits[1])
    output = self.output(output)
    return output
class DQN:
  def __init__(self):
    self.model = DRL()
    self.model.cuda()
    self.loss_function = torch.nn.MSELoss(reduction='mean')
    self.memory = []
    self.action_size = 6
    self.gamma = 0.95
    self.epsilon = 1
    self.epsilon_decay = 0.995
    self.epsilon_min = 0.01
  def remember(self, state, action, reward, next_state, done):
    self.memory.append((state, action, reward, next_state, done))
  def train(self, epochs=20 , batch_size=32):
    for i in range(epochs):
      batch_start = 0
      while batch_start < len(self.memory):
        minibatch = self.memory[batch_start: batch_start + batch_size]
        batch_start += batch_size
        for state, action, reward, next_state, done in minibatch:
          target = reward
          if not done:
            target += self.gamma * self.model(next_state[0]).amax()
          target_f = self.model(state[0])
          target_f_expected = target_f.clone().detach()
          target_f_expected[action] = target
          loss = self.loss_function(target_f, target_f_expected)
          loss.backward()
  def act(self, state):
    if np.random.rand() < self.epsilon:
      return random.randrange(self.action_size)
    pred = self.model(state).detach().numpy
    return np.argmax(pred)
  def save_memory(self, directory):
    torch_dir = os.path.join(directory, "torch_data")
    if not(os.path.exists(torch_dir)):
      os.makedirs(torch_dir, exist_ok=True)
    json_data = []
    state_id = 1
    for state, action, reward, next_state, done in self.memory:
      torch.save(state, os.path.join(torch_dir, f"{state_id}.tensor"))
      json_ = {}
      json_['state'] = state_id
      json_['next_state'] = state_id + 1
      json_['reward'] = reward
      json_['action'] = action
      json_['done'] = done
      state_id += 1
      json_data.append(json_)
    _, _ , _, next_state, _ = self.memory[-1]
    torch.save(state, os.path.join(torch_dir, f"{state_id}.tensor"))
    with open(os.path.join(directory, "evidence.json"), 'w') as f:
      json.dump(json_data, f)
  def load_memory(self, directory):
    torch_dir = os.path.join(directory, "torch_data")
    self.memory = []
    with open(os.path.join(directory, "evidence.json"), 'r') as f:
      json_data = json.load(f)
    for item in json_data:
      state = torch.load(os.path.join(torch_dir, f"{item['state']}.tensor"))
      action = item['action']
      reward = float(item['reward'])
      done = item['done']
      next_state = torch.load(os.path.join(torch_dir, f"{item['next_state']}.tensor"))
      self.memory.append((state, action, reward, next_state, done))

dqn = DQN()