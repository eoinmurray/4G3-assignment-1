import random

R = [0.0, 0.0, 0.0, 0.0, 1.0]
V = [0.0, 0.0, 0.0, 0.0, 0.0]
gamma = 0.9
alpha = 0.1
max_episodes = 1000
max_steps = 20

for episode in range(max_episodes):
    state = random.randint(0, 4)  # start state
    for step in range(max_steps):
      if random.randint(0, 1) == 0:
        next_state = min(state + 1, 4)  # move right, cap at S4
      else:
        next_state = max(state - 1, 0)  # move left, cap at S0
      
      reward = R[next_state]
      V[state] += alpha * (reward + gamma * V[next_state] - V[state])
      state = next_state
      if state == 4:  # episode ends when reaching S4
        break

print("Estimated Values:", V)