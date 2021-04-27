# 
# read this! http://gym.openai.com/
# 

import gym
env = gym.make("CartPole-v1")
observation = env.reset()

number_of_episodes = 1000
for _ in range(number_of_episodes):
    env.render()
    # actions are just represented by an index (integer or list of integers)
    actions = range(env.action_space.n)
    
    #################################
    # you need to pick an action
    #
    import random
    chosen_action = random.choices(actions)[0]
    #
    #
    #################################
    
    # perform the action in the env
    observation, reward, done, info = env.step(chosen_action)

    if done:
        observation = env.reset()
    
    env.close()