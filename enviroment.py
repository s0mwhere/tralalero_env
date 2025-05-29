
import numpy as np
import gymnasium as gym

from user_cal import User
from target_cal import Target
from clutter_cal import Clutter

class base_station(gym.Env):
    def __init__(self, para):
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
        
        self.export_data_flag = 1
    
    def get_obs(self):
        # Get current action parameters
        self.obs_tx_ma_array = np.append(self.tx_ma_array, self.para.rx_posit)
        self.obs_beamform_array = np.array(
            [self.beamform_array.real, self.beamform_array.imag]).reshape(self.para.tx_ma_num * (self.para.comn_usr_num+1) * 2)
        # Get data rates
        array_recieve = []
        for usr in self.userlist:
            array_recieve.append(usr.get_reciev_pow())
        
        #get SCNR
        scnr = np.array((self.SCNR))
        return ({
            "Current Antenna array position":np.array(self.obs_tx_ma_array, dtype=np.double),
            "Current beamforming Power":np.array(self.obs_beamform_array, dtype=np.double),
            "Received signal power at each user":np.array(array_recieve, dtype=np.double),
            "Radar signal-to-clutter-plus-noise ratio":np.array(scnr),
        })

    
    def reset(self, seed=None, options=None):
        self.steep = 0  #for testing only

    #reset action
        self.SCNR = 0 #needed for first obs
        self.split_fact = np.random.uniform(low=0, high=np.pi)
        self.tx_ma_array = self.para.default_tx_ma_array.copy()

        self.beamform_array = (np.random.randn(self.para.tx_ma_num, self.para.comn_usr_num+1) + 1j * np.random.randn(self.para.tx_ma_num, self.para.comn_usr_num+1))*0.13

    #reset entity
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
            if tx_ma_adjust[i] >= 0 and self.tx_ma_array[i] <= self.para.segment_length-0.01: self.tx_ma_array[i] += 0.01
            elif self.tx_ma_array[i]>=0.01: self.tx_ma_array[i] -= 0.01

        #beam forming matrix
        self.beamforming_matrix = np.array(beamforming_real_part + beamforming_img_part*1j).reshape(self.para.tx_ma_num, self.para.comn_usr_num+1)

        for i in range(self.para.tx_ma_num):
            for n in range(self.para.comn_usr_num+1):
                if self.beamforming_matrix[i][n] >= 1: self.beamform_array[i][n] += 0.01
                else: self.beamform_array[i][n] -= 0.01

        #split factor
        if action[-1] >= 0: self.split_fact += 0.01
        else: self.split_fact -= 0.01

        #--update entity (FRVs and channel vect)--
        for i in range(self.para.comn_usr_num):
            self.userlist[i].update(self.para, self.tx_ma_array)

        for i in range(self.para.clutter_num):
            self.clutrlist[i].update(self.para, self.tx_ma_array)
        
        self.target.update(self.para, self.tx_ma_array)

        #--calculate individual data rate for each user--
        self.check = True
        self.data_rate = 0
        for i in range(self.para.comn_usr_num):
            self.data_rate = 0
            self.numer = 0
            self.denom = 0
            for n in range(self.para.comn_usr_num+1):
                if n == i: continue
                self.denom += np.linalg.norm(np.dot((self.userlist[i].channel_vect.conj()), 
                                                       (self.beamform_array[:,n]).reshape(self.para.comn_usr_num+1, 1)))**2
            self.denom = (1-self.split_fact)*self.denom + self.para.variance_watt
            self.numer = (1-self.split_fact)*np.linalg.norm(np.dot((self.userlist[i].channel_vect.conj()), 
                                                       (self.beamform_array[:,i]).reshape(self.para.comn_usr_num+1, 1)))**2
            self.data_rate = np.log2(1+self.numer/self.denom)
            self.userlist[i].set_data_rate(self.data_rate)
            if self.check == True:
                self.numer1 = self.numer
                self.denom1 = self.denom
                self.check = False

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
                                                    (self.beamform_array[:][i]).reshape(self.para.comn_usr_num+1, 1))))**2

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
                                                       (self.beamform_array[:,n]).reshape(self.para.comn_usr_num+1, 1))))**2
            self.userlist[i].set_reciev_pow(self.reciev_pow)
            self.harvest_pow = self.split_fact*self.para.const_v/(1+np.exp(-self.para.const_g*(self.reciev_pow-self.para.const_y)))
            self.userlist[i].set_harvest_pow(self.harvest_pow)

        #--checking stage--
        terminated = False
        truncated = False
        self.reward = 0
        reward = 0
        penalty = 0

        #check beam power threshold
        self.beam_power = np.trace(np.matmul((self.beamform_array.conj().T),(
                            self.beamform_array)))
        self.beam_power = self.beam_power.real
        if self.beam_power > self.para.std_po_watt:
            penalty += self.beam_power - self.para.std_po_watt
            terminated = True
        else:
            reward += 1

        #check SCNR threshold
        if self.SCNR < self.para.SCNR_min_watt:
            penalty += self.para.SCNR_min_watt - self.SCNR
            terminated = True
        else:
            reward += 1

        #check harvest energy threshold for k user
        for usr in self.userlist:
            if usr.get_harvest_pow() < self.para.E_min_watt:
                penalty += self.para.E_min_watt - usr.get_harvest_pow()
                terminated = True
            else:
                reward += 1

        #check coupling tx_ma
        for i in range(self.para.tx_ma_num-1):
            if abs(self.tx_ma_array[i] - self.tx_ma_array[i+1]) < self.para.do_min_dist:
                penalty += 1
                terminated = True
            else:
                reward += 1

        if not terminated:
            reward += self.sum_data_rate*10
        else:
            reward = -penalty*1000

        self.reward = reward.real
        self.terminated = terminated

        observation = self.get_obs()
        info = {}
        
        
        #for testing purpose only
        if(self.export_data_flag == 0):
            self.export_data_matlab()
            self.export_data_flag = 1
        #self.steep += 1
        #if self.steep >= 10: terminated = True
        
        return observation, reward, terminated, truncated, info
    
    def export_data_matlab(self):
        # create file
        data_file = open("enviroment_data.txt", "a")

        # write out data of users
        """
        user content including:
            Distance 
            number of path propagation
            Angle of direction array

        """
        # content
        user_counter = 0
        for usr in self.userlist:
            user_counter += 1
            de_string="usr"+str(user_counter)+"_distance="+str(usr.distance)+";\nusr"+str(user_counter)+"_num_path="+str(self.para.channel_path_num)+";\nusr"+str(user_counter)+"_AoD=["
            for angle in usr.angle:
                de_string += str(angle)
                if(angle!= usr.angle[-1]):
                    de_string += ','
            de_string +="];\nusr"+str(user_counter)+"_path_gain_distribution=["
            for gain in usr.path_gain:
                de_string += str(gain)
                if(gain!= usr.path_gain[-1]):
                    de_string+=','
            de_string +="];\n"
            data_file.write(de_string)
            break           


        de_string = "targ1_AoD=" + str(self.target.angle)+';\n'
        de_string +="targ1_atten_coeff=" + str(self.target.atten_coeff)+';\n'
        de_string +="targ1_doppler_freq=" + str(self.target.atten_coeff)+';\n'
        de_string +="targ1_rx_posit=" + str(self.target.rx_posit)+';\n'
        de_string +="sampling_period =" + str(self.para.sampling_period)+';\n'
        de_string +='clut_channel_modl=['
        for i in range(2):
            for n in self.clutrlist[i].channel_modl:
                de_string += str(n)+','
            de_string += ';'
        de_string += '];\n'
        data_file.write(de_string)

        

        # export system, data
        """
        System data including:
            tx antenna array
            beamforming matrix

        """
        
        action_string = "antenna_array=["
        for act in self.tx_ma_array:
            action_string += str(act)
            if(act!=self.tx_ma_array[-1]):
                action_string+=','
        action_string += "];\nbeam_matrix =["
        for m in self.beamform_array:
            action_string += ';'
            for n in m:
                action_string += str(n)
                if(n != self.beamform_array[-1][-1]):
                    action_string += ','
        action_string += '];\n'+'split_fact = '+str(self.split_fact)+';\n\n'
        data_file.write(action_string)

        """
        calculated datas:
            Max channel gain
            Array response vector 
            Channel vector

            sum data rate
        """
        for usr in self.userlist:
            de_string = ("Max Channel gain="+str(usr.path_gain_var)+
                            "\nUser array response vector: \n"+str(usr.FRVs_usr)+
                            "\nChannel Array: "+str(usr.channel_vect)+
                            "\ndenom: "+str(self.denom1)+
                            "\nnumer: "+str(self.numer1)+
                            "\ndata rate: "+str(usr.data_rate)+'\nSCNR: '+str(self.SCNR))
            data_file.write(de_string)
            break
        
        data_file.write("\ncal Beam Power: "+str(self.beam_power))
        data_file.close()
