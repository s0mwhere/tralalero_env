import numpy as np

class Clutter:

    def __init__(self, para):
        self.distance = np.random.uniform(low=para.dist_min, high=para.dist_max)
        self.speed = np.random.uniform(low=0, high=20)
        self.angle = np.random.uniform(low=0, high=np.pi)
        self.rx_posit = para.rx_posit
        self.FRVs_clutr = np.zeros((para.tx_ma_num),dtype=np.complex128)

        self.doppler_freq = (2 * self.speed * para.carrier_freq) / para.light_spd
        self.atten_coeff = np.sqrt((para.lamda**2 * para.RCS)/((4*np.pi)**3 * self.distance*4))

    def set_FRVs_clutr(self, para, tx_ma_array):
        for col in range(para.tx_ma_num):
            self.FRVs_clutr[col] = np.exp(1j * (2 * np.pi / para.lamda) * tx_ma_array[col] * np.cos(self.angle))
            #untranspose

    def set_channel_modl(self, para):
        self.channel_modl = self.atten_coeff*np.exp(1j*2*np.pi*self.doppler_freq*para.sampling_period)*np.exp(
            1j*(2*np.pi/para.lamda)*self.rx_posit*np.cos(self.angle))*self.FRVs_clutr
        #untranspose

    def update(self, para, tx_ma_array):
        self.set_FRVs_clutr(para, tx_ma_array)
        self.set_channel_modl(para)