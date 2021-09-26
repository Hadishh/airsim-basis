

from typing import Collection


def interpret_action(id, car_controls):
    car_controls.brake = 0
    car_controls.throttle = 0.5
    if id == 0:
        car_controls.throttle = 0
        car_controls.brake = 1
    elif id == 1:
        car_controls.steering = 0
    elif id == 2:
        car_controls.steering = 0.5
    elif id == 3:
        car_controls.steering = -0.5
    elif id == 4:
        car_controls.steering = 0.25
    else:
        car_controls.steering = -0.25
    return car_controls

def compute_reward(car_state, collision_info):
    MAX_SPEED = 10
    MIN_SPEED = 1
    done = False
    speed_reward = (car_state.speed - MIN_SPEED) / (MAX_SPEED - MIN_SPEED)
    if speed_reward > 1:
        # penalise the high speed
        speed_reward *= 0.01
    elif speed_reward < 0:
        speed_reward = 0.01
    collision_reward = 10
    if collision_info.has_collided:
        collision_reward = -float("inf") 
    if (collision_reward < 0):
        done = True
    return speed_reward * 100 + collision_reward, done