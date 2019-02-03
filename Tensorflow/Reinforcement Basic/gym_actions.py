## ACTIONS IN OPEN AI GYM**
# Prashant Brahmbhatt

import gym
env = gym.make('CartPole-v0')

#print(env.action_space)   # outputs 'Discrete(2)' which denotes two sets of actions 0 or 1

#print(env.observation_space) 
# # which outputs 'Box(4,)' so it gives out 4 observations which are
# cart position, cart velociy, pole angle, and angular velocity as described in the documentation
observation = env.reset()

for t in range(1000):

    env.render()

    cart_pos, cart_vel, pole_ang, ang_vel = observation

    # Now we try to balance the cartpole by measuring the angle
    # if the angle if greater than 0 then move the cart to the right
    # else move the cart to the left

    if pole_ang > 0:
        action = 1
    else:
        action = 0

    # Now performing the action
    observation, reward, done, info = env.step(action) 
