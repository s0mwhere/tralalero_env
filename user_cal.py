import numpy as np

class User:

    def __init__(self, para):
        self.distance = np.random.uniform(low=para.dist_min, high=para.dist_max)

        self.angle = np.random.uniform(low=0, high=np.pi, size=(para.channel_path_num))
        self.FRVs_usr = np.zeros((para.channel_path_num, para.tx_ma_num),dtype=np.complex128)
        self.path_gain_var = para.path_loss_ref*(self.distance**(-para.path_loss_exp))/para.channel_path_num
        self.path_gain = np.zeros((para.channel_path_num,),dtype=np.complex128)
        self.channel_vect = np.zeros((para.tx_ma_num),dtype=np.complex128)
        self.AWGN = np.sqrt(para.variance_watt/2) * (np.random.randn() + 1j * np.random.randn())

        self.reciev_pow = 0

        for i in range(para.channel_path_num):
            self.path_gain[i] = np.sqrt(self.path_gain_var/2) * (np.random.randn() + 1j * np.random.randn())
    
    def set_FRVs_usr(self, para, tx_ma_array):
        for row in range(para.channel_path_num):
            for col in range(para.tx_ma_num):
                self.FRVs_usr[row][col] = np.exp(1j * (2 * np.pi / para.lamda) * tx_ma_array[col] * np.cos(self.angle[row]))
                #untranspose

    def set_channel_vect(self, para):    
        for i in range(para.channel_path_num):
            self.channel_vect += self.path_gain[i]*self.FRVs_usr[i]
            #untranspose
    
    def update(self, para, tx_ma_array):
        self.set_FRVs_usr(para, tx_ma_array)
        self.set_channel_vect(para)
        
    def set_data_rate(self, data_rate):
        self.data_rate = data_rate
        pass

    def get_data_rate(self):
        return self.data_rate
    
    def set_reciev_pow(self, pow):
        self.reciev_pow = pow

    def get_reciev_pow(self):
        return self.reciev_pow
    
    def set_harvest_pow(self, pow):
        self.harvest_pow = pow

    def get_harvest_pow(self):
        return self.harvest_pow
