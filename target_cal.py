import numpy as np

class Target:

    def __init__(self, para):
        self.distance = para.dist_target
        self.angle = para.angle_target
        self.speed = para.speed_target
        self.rx_posit = para.rx_posit

        self.FRVs_targ = np.zeros((para.tx_ma_num),dtype=np.complex128)

        for col in range(para.tx_ma_num):
            self.FRVs_targ[col] = np.exp(1j * (2 * np.pi / para.lamda) * para.tx_ma_array[col] * np.cos(self.angle))
            #untranspose

        self.doppler_freq = (2 * self.speed * para.carrier_freq) / para.light_spd
        self.atten_coeff = np.sqrt((para.lamda**2 * para.RCS)/((4*np.pi)**3 * self.distance*4))

        self.channel_modl = self.atten_coeff*np.exp(1j*2*np.pi*self.doppler_freq*para.sampling_period)*np.exp(
            1j*(2*np.pi/para.lamda)*self.rx_posit*np.cos(self.angle))*self.FRVs_targ
        #untranspose

'''m=Target()
print(m.atten_coeff)'''