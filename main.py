from train import train, env, select_action

# # Train the RL Model
# train()

# Testing the trained RL Model
done = False
cnt = 0
observation = env.reset()
while True:
    cnt += 1
    env.render()
    action = select_action(observation)
    observation, reward, done, _ = env.step(action)

print(f"Game lasted {cnt} moves")