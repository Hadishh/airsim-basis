
import torch
import numpy as np

torch.cuda.init()

from DeepNetworks.inceptionv3_based_model import dqn
dqn.model.cuda()
dqn.load_memory("observed_data")
dqn.train(epochs=10, batch_size=32)