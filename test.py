import gymnasium as gym
import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from enviroment import base_station

env = base_station()

'''episodes = 50
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
		print('next step')'''

#--loging and save model
logdir = "logs"
models_dir = "models/PPO_1"                  # output model
model_path = "models/PPO_1/best_model.zip"     # intput model
if not os.path.exists(logdir):
    os.makedirs(logdir)

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

eval_callback = EvalCallback(env, best_model_save_path=models_dir,
                             eval_freq=1000,
                             deterministic=True, render=False)

model = PPO('MultiInputPolicy', env, verbose=1, tensorboard_log=logdir, ent_coef=0.01, gamma=0.95)
#model = PPO.load(model_path, env=env)

TIMESTEPS = 500000
model.learn(total_timesteps=TIMESTEPS, callback=eval_callback, tb_log_name="PPO")

env.close()