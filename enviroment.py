
import numpy as np
import gymnasium as gym

from parameter import Para
from user_cal import User
from target_cal import Target
from clutter_cal import Clutter

para = Para()

class base_station(gym.Env):
    def __init__(self):
        super().__init__()
        self.para = para
        
        self.observation_space = gym.spaces.Dict({
            "Current Antenna array position":gym.spaces.Box(0,self.para.segment_length,shape=(self.para.tx_ma_num+1,),dtype=np.double),
            "Current beamforming Power":gym.spaces.Box(-1,1,shape=(self.para.tx_ma_num*(self.para.comn_usr_num+1)*2,),dtype=np.double),
            "Received signal power at each user":gym.spaces.Box(0,40,shape=(self.para.comn_usr_num,),dtype=np.double),
            "Radar signal-to-clutter-plus-noise ratio":gym.spaces.Box(0,40, shape=(1,),dtype=np.double),
        })

        self.action_space = gym.spaces.Box(low=-1,high=1,
                                           shape=((self.para.tx_ma_num+
                                                   (self.para.tx_ma_num * (self.para.comn_usr_num+1)) * 2+1),))
    
    def get_obs(self):
        # Get current action parameters
        self.obs_tx_ma_array = np.append(self.para.tx_ma_array, self.para.rx_posit)
        self.obs_beamform_array = np.array(
            [self.beamform_array.real, self.beamform_array.imag]).reshape(self.para.tx_ma_num * (self.para.comn_usr_num+1) * 2)
        # Get data rates
        array_recieve = []
        for usr in self.userlist:
            array_recieve.append(usr.get_reciev_pow())
        scnr = np.array((self.SCNR))
        return ({
            "Current Antenna array position":np.array(self.obs_tx_ma_array, dtype=np.double),
            "Current beamforming Power":np.array(self.obs_beamform_array, dtype=np.double),
            "Received signal power at each user":np.array(array_recieve, dtype=np.double),
            "Radar signal-to-clutter-plus-noise ratio":np.array(scnr),
        })

    
    def reset(self, seed=None, options=None):
        self.terminated = False
        self.truncated = False
        self.steep = 0
        self.SCNR = 0

        self.split_fact = np.random.uniform(low=0, high=np.pi)
        self.para.tx_ma_array = self.para.default_tx_ma_array.copy()

        self.beamform_array = np.random.randn(self.para.tx_ma_num, self.para.comn_usr_num+1) + 1j * np.random.randn(self.para.tx_ma_num, self.para.comn_usr_num+1)
        self.beamform_array = self.beamform_array / np.linalg.norm(self.beamform_array, 'fro') * np.sqrt(self.para.std_po_watt)

        self.userlist = []
        for _ in range(self.para.comn_usr_num):
            self.userlist.append(User(self.para))

        self.clutrlist = []
        for _ in range(self.para.clutter_num):
            self.clutrlist.append(Clutter(self.para))
        
        self.target = Target(self.para)

        observation = self.get_obs()
        info = {}
        return observation, info

    def step(self, action):
        #--register action--
        tx_ma_adjust = action[:self.para.tx_ma_num]
        
        buffer_array = action[self.para.tx_ma_num:-1]*0.13 # Extract and Adjust beamforming element (change to approviate normalization later)
        beamforming_real_part = buffer_array[:self.para.tx_ma_num*(self.para.comn_usr_num+1)]
        beamforming_img_part = buffer_array[self.para.tx_ma_num*(self.para.comn_usr_num+1):]

        #---modify and apply action--
        #antena array
        for i in range(self.para.tx_ma_num):
            if tx_ma_adjust[i] >= 0 and self.para.tx_ma_array[i] <= self.para.segment_length-0.01: self.para.tx_ma_array[i] += 0.01
            elif self.para.tx_ma_array[i]>=0.01: self.para.tx_ma_array[i] -= 0.01

        #beam forming matrix
        self.beamforming_matrix = np.array(beamforming_real_part + beamforming_img_part*1j).reshape(self.para.tx_ma_num, self.para.comn_usr_num+1)

        for i in range(self.para.tx_ma_num):
            for n in range(self.para.comn_usr_num+1):
                if self.beamforming_matrix[i][n] >= 1: self.beamform_array[i][n] += 0.01
                else: self.beamform_array[i][n] -= 0.01

        #split factor
        if action[-1] >= 0: self.split_fact += 0.01
        else: self.split_fact -= 0.01

        #--update entity--
        for i in range(self.para.comn_usr_num):
            self.userlist[i].update(self.para)

        for i in range(self.para.clutter_num):
            self.clutrlist[i].update(self.para)
        
        self.target.update(self.para)

        #--calculate individual data rate for each user--
        self.data_rate = 0
        for i in range(self.para.comn_usr_num):
            self.data_rate = 0
            for n in range(self.para.comn_usr_num+1):
                if n == i: continue
                self.data_rate += (np.linalg.norm(np.dot((self.userlist[i].channel_vect.conj()), 
                                                       (self.beamform_array[:][n]).reshape(self.para.comn_usr_num+1, 1))))**2
            self.data_rate = (1-self.split_fact)*self.data_rate + self.para.variance_watt
            self.data_rate = np.log2(1+((1-self.split_fact)*(np.linalg.norm(np.dot((self.userlist[i].channel_vect.conj()), 
                                                       (self.beamform_array[:][i]).reshape(self.para.comn_usr_num+1, 1))))**2)/self.data_rate)
            self.userlist[i].set_data_rate(self.data_rate)

        #data rate sum
        self.sum_data_rate = 0
        for usr in self.userlist:
            self.sum_data_rate += usr.get_data_rate()
        
        #SCNR
        self.SCNR = 0
        self.SCNR_denom1 = 0
        self.SCNR_denom2 = 0

        self.SCNR_numer = (np.linalg.norm(np.dot((self.target.channel_modl.conj()), 
                                                    (self.beamform_array[:][self.para.comn_usr_num]).reshape(self.para.comn_usr_num+1, 1))))**2
                                                                        #k+1 -> k

        for i in range(self.para.comn_usr_num):
            self.SCNR_denom1 += (np.linalg.norm(np.dot((self.target.channel_modl.conj()), 
                                                    (self.beamform_array[:][n]).reshape(self.para.comn_usr_num+1, 1))))**2

        for i in range(self.para.clutter_num):
            for n in range(self.para.comn_usr_num+1):
                self.SCNR_denom2 += (np.linalg.norm(np.dot((self.clutrlist[i].channel_modl.conj()), 
                                                    (self.beamform_array[:][n]).reshape(self.para.comn_usr_num+1, 1))))**2
                
        self.SCNR = self.SCNR_numer/(self.SCNR_denom1+self.SCNR_denom2+self.para.variance_watt)

        # harvested energy and received power at user k
        self.reciev_pow = 0
        self.harvest_pow = 0
        for i in range(self.para.comn_usr_num):
            self.reciev_pow = 0
            for n in range(self.para.comn_usr_num+1):
                self.reciev_pow += (np.linalg.norm(np.dot((self.userlist[i].channel_vect.conj()), 
                                                       (self.beamform_array[:][n]).reshape(self.para.comn_usr_num+1, 1))))**2
            self.userlist[i].set_reciev_pow(self.reciev_pow)
            self.harvest_pow = self.split_fact*self.para.const_v/(1+np.exp(-self.para.const_g*(self.reciev_pow-self.para.const_y)))
            self.userlist[i].set_harvest_pow(self.harvest_pow)

        #--checking stage--
        terminated = False
        truncated = False
        reward = 0

        #check beam power threshold
        self.beam_power = np.trace(np.matmul((self.beamform_array.conj().T),(
                            self.beamform_array)))
        if self.beam_power > self.para.std_po_watt:
            reward -= self.beam_power - self.para.std_po_watt
            terminated = True

        #check SCNR threshold
        if self.SCNR < self.para.SCNR_min_watt:
            reward -= self.para.SCNR_min_watt - self.SCNR
            terminated = True

        #check harvest energy threshold for k user
        for usr in self.userlist:
            if usr.get_harvest_pow() < self.para.E_min_watt:
                reward -= self.para.E_min_watt - usr.get_harvest_pow()
                terminated = True

        #check coupling tx_ma
        for i in range(self.para.tx_ma_num-1):
            if abs(self.para.tx_ma_array[i] - self.para.tx_ma_array[i+1]) < self.para.do_min_dist:
                reward -= 1
                terminated = True

        if not terminated:
            reward += self.sum_data_rate

        self.reward = reward
        self.terminated = terminated

        observation = self.get_obs()
        info = {}
        

        #for testing purpose only
        self.steep += 1
        if self.steep >= 10: terminated = True
        
        return observation, reward, terminated, truncated, info
