import numpy as np
from parameter import Para
from user_cal import User
from target_cal import Target
from clutter_cal import Clutter

para = Para()

class base_station:
    def __init__(self):
        self.split_fact = np.random.uniform(low=0, high=np.pi)
        

        self.tx_ma_array = para.tx_ma_array

        self.beamform_array = np.random.randn(para.tx_ma_num, para.comn_usr_num+1) + 1j * np.random.randn(para.tx_ma_num, para.comn_usr_num+1)
        self.beamform_array = self.beamform_array / np.linalg.norm(self.beamform_array, 'fro') * np.sqrt(para.std_po_watt)

        self.userlist = []
        for i in range(para.comn_usr_num):
            self.userlist.append(User(para))

        self.clutrlist = []
        for i in range(para.clutter_num):
            self.clutrlist.append(Clutter(para))
        
        self.target = Target(para)

        self.data_rate = 0
        for i in range(para.comn_usr_num):
            self.data_rate = 0
            for n in range(para.comn_usr_num+1):
                if n == i: continue
                self.data_rate += (np.linalg.norm(np.dot((self.userlist[i].channel_vect.conj()), 
                                                       (self.beamform_array[:][n]).reshape(para.comn_usr_num+1, 1))))**2
            self.data_rate = (1-self.split_fact)*self.data_rate + para.variance_watt
            self.data_rate = np.log2(1+((1-self.split_fact)*(np.linalg.norm(np.dot((self.userlist[i].channel_vect.conj()), 
                                                       (self.beamform_array[:][i]).reshape(para.comn_usr_num+1, 1))))**2)/self.data_rate)
            self.userlist[i].set_data_rate(self.data_rate)
        
        self.SCNR = 0
        self.SCNR_denom1 = 0
        self.SCNR_denom2 = 0

        self.SCNR_numer = (np.linalg.norm(np.dot((self.target.channel_modl.conj()), 
                                                    (self.beamform_array[:][para.comn_usr_num]).reshape(para.comn_usr_num+1, 1))))**2
                                                                        #k+1 -> k

        for i in range(para.comn_usr_num):
            self.SCNR_denom1 += (np.linalg.norm(np.dot((self.target.channel_modl.conj()), 
                                                    (self.beamform_array[:][n]).reshape(para.comn_usr_num+1, 1))))**2

        for i in range(para.clutter_num):
            for n in range(para.comn_usr_num+1):
                self.SCNR_denom2 += (np.linalg.norm(np.dot((self.clutrlist[i].channel_modl.conj()), 
                                                    (self.beamform_array[:][n]).reshape(para.comn_usr_num+1, 1))))**2
                
        self.SCNR = self.SCNR_numer/(self.SCNR_denom1+self.SCNR_denom2+para.variance_watt)

        self.reciev_pow = 0
        self.harvest_pow = 0
        for i in range(para.comn_usr_num):
            self.reciev_pow = 0
            for n in range(para.comn_usr_num+1):
                self.reciev_pow += (np.linalg.norm(np.dot((self.userlist[i].channel_vect.conj()), 
                                                       (self.beamform_array[:][n]).reshape(para.comn_usr_num+1, 1))))**2
            self.userlist[i].set_reciev_pow(self.reciev_pow)
            self.harvest_pow = self.split_fact*para.const_v/(1+np.exp(-para.const_g*(self.reciev_pow-para.const_y)))
            self.userlist[i].set_harvest_pow(self.harvest_pow)

        self.total_data_rate = 0
        for i in self.userlist:
            self.total_data_rate += i.get_harvest_pow()

'''m = base_station()
print(m.SCNR)'''