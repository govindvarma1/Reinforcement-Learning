import gymnasium as gym
import numpy as np
import pickle as pkl

cliffEnv = gym.make('CliffWalking-v0')

q_table = np.zeros((cliffEnv.observation_space.n, cliffEnv.action_space.n))

def policy(state, explore_rate=0.0):
    action = int(np.argmax(q_table[state]))
    if np.random.random() <= explore_rate:
        action = int(np.random.randint(low=0, high=cliffEnv.action_space.n, size=1))
    return action

EPSILON = 0.1
ALPHA = 0.1
GAMMA = 0.9
NUM_EPISODES = 1000

for episode in range(NUM_EPISODES):
    done = False
    total_reward = 0
    episode_length = 0
    state, _ = cliffEnv.reset()  # Ensure state is correctly unpacked
    action = policy(state, EPSILON)
    
    while not done:
        next_state, reward, done, _, _ = cliffEnv.step(action)
        next_action = policy(next_state, EPSILON)
        
        q_table[state, action] += ALPHA * (reward + GAMMA * q_table[next_state, next_action] - q_table[state, action])
        
        state = next_state
        action = next_action

        total_reward += reward
        episode_length += 1
    print('Episode:', episode, 'Episode Length: ', episode_length, 'Total Reward:', total_reward)

done = False
state, _ = cliffEnv.reset()  # Ensure state is correctly unpacked
action = policy(state, EPSILON)

while not done:
    next_state, reward, done, _, _ = cliffEnv.step(action)
    next_action = policy(next_state, EPSILON)
    
    q_table[state, action] += ALPHA * (reward + GAMMA * q_table[next_state, next_action] - q_table[state, action])
    
    state = next_state
    action = next_action

cliffEnv.close()

pkl.dump(q_table, open('./cliff walking/sarsa_q_table.pkl', 'wb'))
