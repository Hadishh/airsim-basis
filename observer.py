import airsim
from utils import interpret_action, compute_reward
import os
import cv2
import torch
import numpy as np

torch.cuda.init()

from DeepNetworks.inceptionv3_based_model import dqn
client = airsim.CarClient()
client.confirmConnection()
client.enableApiControl(True)
import time
curr_state_img = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])
curr_state_img = curr_state_img[0]
image = airsim.string_to_uint8_array(curr_state_img.image_data_uint8)
image = image.reshape(144, 256, 3)
# res = upload(response., image_data_uint8)
curr_state = np.resize(image, (299, 299, 3))
curr_state_tensor = torch.tensor(curr_state).cuda()
curr_state_tensor = curr_state_tensor.permute(2, 0, 1)
curr_car_state = client.getCarState()
save_after = 2000
iter = 0
car_stopped_iters = 0
dqn.model.cuda()
while True:
    start = time.time()
    action = dqn.act(curr_state_tensor)
    controls = client.getCarControls()
    controls = interpret_action(action, controls)
    client.setCarControls(controls)
    next_car_state = client.getCarState()
    if(next_car_state.speed < 0.01 and curr_car_state.speed < 0.01):
        car_stopped_iters += 1
    else:
        car_stopped_iters = 0
    collisions = client.simGetCollisionInfo()
    next_state_img = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])[0]
    next_state_array = airsim.string_to_uint8_array(next_state_img.image_data_uint8)
    next_state_array = next_state_array.reshape(144, 256, 3)
    next_state_array = np.resize(next_state_array, (299, 299, 3))
    next_state_tensor = torch.tensor(next_state_array).cuda()
    next_state_tensor = next_state_tensor.permute(2, 0, 1)  
    reward, done = compute_reward(next_car_state, collisions)
    dqn.remember((curr_state_tensor, curr_car_state), action, reward, (next_state_tensor, next_car_state), done)
    # print(f"took {time.time() - start} with action {action}")
    if (done or car_stopped_iters == 100):
        client.reset()
        curr_state_img = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])
        curr_state_img = curr_state_img[0]
        image = airsim.string_to_uint8_array(curr_state_img.image_data_uint8)
        image = image.reshape(144, 256, 3)
        # res = upload(response., image_data_uint8)
        curr_state = np.resize(image, (299, 299, 3))
        curr_state_tensor = torch.tensor(curr_state).cuda()
        curr_state_tensor = curr_state_tensor.permute(2, 0, 1)
        curr_car_state = client.getCarState()
    else:
        curr_state_tensor = next_state_tensor
        curr_car_state = next_car_state
    iter += 1
    if (iter % 100 == 0):
        print(f"passed {iter} iters")
    if(iter % save_after == 0):
        client.reset()
        dqn.save_memory("data")
    