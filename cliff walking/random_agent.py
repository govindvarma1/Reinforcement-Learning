import gymnasium as gym
import numpy as np

# Create the CliffWalking environment with human render mode
cliffEnv = gym.make('CliffWalking-v0', render_mode='human')

done = False
state = cliffEnv.reset()

while not done:
    # Select a random action from the action space
    action = int(np.random.randint(0, cliffEnv.action_space.n, 1))
    print(state, '--->', action)
    
    # Take the action and get the next state, reward, and done flag
    next_state, reward, done, _, _ = cliffEnv.step(action)
    
    # Update the current state to the next state
    state = next_state

# Close the environment
cliffEnv.close()