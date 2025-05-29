import gymnasium as gym
import numpy as np
import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from enviroment import base_station
from parameter import Para

para = Para()
env = base_station(para)

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        self.best_mean_reward = -np.inf

    def _on_training_start(self) -> None:
        # setup logging general information
        
        pass

    def _on_rollout_start(self) -> None:
        # Set up to log the best perfomance run
        self.best_reward_run = 0
        self.best_data_rate_run = 0
        self.step_counter = 0
        self.reward = 0
        self.sum_reward = 0
        self.data_rate = 0
        self.sum_data_rate = 0
        self.log_antenna_array = []
        self.best_antenna_array_0 = 0
        self.best_antenna_array_1 = 0
        self.best_antenna_array_2 = 0
        self.best_antenna_array_3 = 0
        self.beam_power = 0
        self.best_beamforming_power = 0
        
    def _on_step(self) -> bool:
        self.step_counter += 1
        
        # log other information while traning
        # beam power
        self.beam_power = self.training_env.get_attr('beam_power')[-1]
        # antenna array distribution
        self.log_antenna_array = self.training_env.get_attr('tx_ma_array')[-1]
        # sum data rate
        self.data_rate = self.training_env.get_attr('sum_data_rate')[-1]
        self.sum_data_rate += self.data_rate
        
        self.reward = self.training_env.get_attr('reward')[-1]
        self.sum_reward += self.reward
        
        # log step by step
        self.logger.record('step/reward', self.reward)
        self.logger.record('step/data_rate', self.data_rate)
        self.logger.record('step/beam_power', self.beam_power)
        self.logger.record('step/antenna_array_1', self.log_antenna_array[0])
        self.logger.record('step/antenna_array_2', self.log_antenna_array[1])
        self.logger.record('step/antenna_array_3', self.log_antenna_array[2])
        self.logger.record('step/antenna_array_4', self.log_antenna_array[3])
        # check the best performance run
        if self.best_reward_run < self.reward:
            self.best_reward_run = self.reward
            # save the best model parameters
            # save antenna array parameters
            self.best_data_rate_run = self.data_rate
            self.best_antenna_array_0 = self.log_antenna_array[0]
            self.best_antenna_array_1 = self.log_antenna_array[1]
            self.best_antenna_array_2 = self.log_antenna_array[2]
            self.best_antenna_array_3 = self.log_antenna_array[3]
            # save beamforming power
            self.best_beamforming_power = self.training_env.get_attr('beam_power')[-1]
        
        return True

    def _on_rollout_end(self) -> None:
        # Calculate the mean reward
        self.mean_reward = self.sum_reward / self.step_counter
        # calculate the mean data rate
        self.mean_data_rate = self.sum_data_rate / self.step_counter
        self.logger.record('rollout_custom/total_num_step', self.step_counter)
        self.logger.record('rollout_custom/mean_reward', self.mean_reward)
        self.logger.record('rollout_custom/mean_data_rate', self.mean_data_rate)
        self.logger.record('rollout_custom/best_reward_run', self.best_reward_run)
        self.logger.record('rollout_custom/best_data_rate_run', self.best_data_rate_run)
        self.logger.record('rollout_custom/best_antenna_array_0', self.best_antenna_array_0)
        self.logger.record('rollout_custom/best_antenna_array_1', self.best_antenna_array_1)
        self.logger.record('rollout_custom/best_antenna_array_2', self.best_antenna_array_2)
        self.logger.record('rollout_custom/best_antenna_array_3', self.best_antenna_array_3)
        self.logger.record('rollout_custom/best_beamforming_power', self.best_beamforming_power)
    

    # def _on_training_end(self) -> None:

#--loging and save model
logdir = "logs"
models_dir = "models/PPO_1"                  # output model
model_path = "models/PPO_1/best_model.zip"     # intput model
if not os.path.exists(logdir):
    os.makedirs(logdir)

'''if not os.path.exists(models_dir):
    os.makedirs(models_dir)'''

eval_callback = EvalCallback(env, best_model_save_path=models_dir,
                             eval_freq=1000,
                             deterministic=True, render=False)

model = PPO('MultiInputPolicy', env, verbose=1, tensorboard_log=logdir, ent_coef=0.01, gamma=0.95)
#model = PPO.load(model_path, env=env)

TIMESTEPS = 100000
model.learn(total_timesteps=TIMESTEPS, callback=TensorboardCallback(), tb_log_name="PPO")

env.close()