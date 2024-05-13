import gymnasium as gym	

env = gym.make("LunarLander-v2", render_mode="human")

for i_episode in range(20):	
    observation = env.reset()	

    for t in range(100):	
        env.render()	
        print(observation)	

        action = env.action_space.sample()	
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            break 

env.close()
