
import numpy as np
import gymnasium as gym

from parameter import Para
from user_cal import User
from target_cal import Target
from clutter_cal import Clutter

para = Para()

class base_station(gym.Env):
    def __init__(self):
        super.__init__()
        self.para = para
        

        self.observation_space = gym.spaces.Dict({
            "Current Antenna array position":gym.spaces.Box(0,self.para.segment_length,shape=(self.para.tx_ma_num,),dtype=np.double),
            "Current beamforming Power":gym.spaces.Box(-1,1,shape=(self.para.tx_ma_num*(self.para.comn_usr_num+1)*2,),dtype=np.double),
            "Received signal power at each user":gym.spaces.Box(0,40,shape=(self.para.comn_usr_num,),dtype=np.double),
            "Radar signal-to-clutter-plus-noise ratio":gym.spaces.Box(0,40, shape=(1,),dtype=np.double),
        })

        self.action_space = gym.spaces.Box(low=-1,high=1,
                                           shape=((self.para.tx_ma_num+
                                                   (self.para.tx_ma_num * (self.para.comn_usr_num+1)) * 2+1),))
    
    def reset(self, seed=None, options=None):
        self.split_fact = np.random.uniform(low=0, high=np.pi)
        self.tx_ma_array = self.para.tx_ma_array

        self.beamform_array = np.random.randn(self.para.tx_ma_num, self.para.comn_usr_num+1) + 1j * np.random.randn(self.para.tx_ma_num, self.para.comn_usr_num+1)
        self.beamform_array = self.beamform_array / np.linalg.norm(self.beamform_array, 'fro') * np.sqrt(self.para.std_po_watt)

        self.userlist = []
        for i in range(self.para.comn_usr_num):
            self.userlist.append(User(self.para))

        self.clutrlist = []
        for i in range(self.para.clutter_num):
            self.clutrlist.append(Clutter(self.para))
        
        self.target = Target(self.para)

    def step(self, action):

        tx_ma_adjust = action[:self.para.tx_ma_num]
        
        buffer_array = action[self.para.tx_ma_num:-1]*0.13 # Extract and Adjust beamforming element
        beamforming_real_part = buffer_array[:self.para.tx_ma_num*(self.para.comn_usr_num+1)]
        beamforming_img_part = buffer_array[self.para.tx_ma_num*(self.para.comn_usr_num+1):]

        # apply action
        for i in range(self.para.tx_ma_num):
            if tx_ma_adjust[i] >= 0: self.para.tx_ma_array[i] += 0.01
            else: self.para.tx_ma_array[i] -= 0.01

        self.beamforming_matrix = np.array(beamforming_real_part + beamforming_img_part*1j).reshape(self.cfg.Number_of_comm_messages,self.Number_of_tx_antenna_elements)

        for i in range(self.para.tx_ma_num):
            for n in range(self.para.comn_usr_num+1):
                if self.beamforming_matrix[i][n] >= 1: self.beamform_array[i][n] += 0.01
                else: self.beamform_array[i][n] -= 0.01

        if action[-1] >= 0: self.split_fact += 0.01
        else: self.split_fact -= 0.01

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

m = base_station()
print(m.tx_ma_array)