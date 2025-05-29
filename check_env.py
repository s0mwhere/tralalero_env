from enviroment import base_station
import gymnasium as gym
from parameter import Para

para = Para()

#env = gym.make("LunarLander-v3", render_mode="human")
env = base_station(para)

episodes = 50
for episode in range(episodes):
	terminated =False
	truncated = False
	obs, info = env.reset()
	while not terminated and not truncated:
		random_action = env.action_space.sample()
		#print("action",random_action)
		#print(env.step(random_action))
		obs, reward, terminated, truncated, info = env.step(random_action)
		print('obs', env.obs_tx_ma_array)
		print('rw', env.reward)
		print('terminated', env.terminated)
		print('next step')
		