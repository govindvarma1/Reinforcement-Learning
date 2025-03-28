import gymnasium as gym
import numpy as np
import pickle as pkl

# Create the CliffWalking environment with human render mode
cliffEnv = gym.make('CliffWalking-v0', render_mode='human')

# Load the pre-trained Q-table
q_table = pkl.load(open('sarsa_q_table.pkl', 'rb'))

def policy(state, explore_rate=0.0):
    action = int(np.argmax(q_table[state]))
    if np.random.random() <= explore_rate:
        action = int(np.random.randint(low=0, high=cliffEnv.action_space.n, size=1))
    return action


NUM_EPISODES = 5

for episode in range(NUM_EPISODES):
    state, _ = cliffEnv.reset()
    done = False
    while not done:
        action = policy(state)
        print(state, '--->', action)
        
        next_state, reward, done, _, _ = cliffEnv.step(action)
        
        state = next_state

# Close the environment
cliffEnv.close()