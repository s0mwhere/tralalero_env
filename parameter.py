import numpy as np

def dbm_watt(n):
        return 10 ** ((n - 30) / 10)

class Para:
    comn_usr_num = 3
    tx_ma_num = 4
    lamda = 0.1
    segment_length = 10*lamda
    do_min_dist = lamda/2

    tx_ma_array = np.zeros((tx_ma_num,))
    for i in range(tx_ma_num):
        tx_ma_array[i]=(segment_length/tx_ma_num)*i

    beamform_array = np.zeros((tx_ma_num, comn_usr_num + 1), dtype=np.complex128)
    rx_posit = np.random.uniform(low=0, high=segment_length)

    angle_target = np.pi/3
    dist_target = 30
    speed_target = 10

    sampling_period = 1e-6

    path_loss_exp = 2.8
    path_loss_ref = 1
    RCS = 2.2

    clutter_num = 2

    const_y = 8
    const_v = 20
    const_g = 0.3
    const_y2 = 1/(1+np.exp(const_y*const_g))

    E_min = -20 #dBm

    d_max = 100
    d_min = 25

    carrier_freq = 2.4e9 #Hz

    channel_path_num = 13

    dist_min = 25
    dist_max = 100

    variance_dbm = -80 #dBm
    std_po_dbm = 32    #dBm
    SCNR_min_dbm = 5   #dBm

    variance_watt = dbm_watt(variance_dbm)
    std_po_watt = dbm_watt(std_po_dbm)
    light_spd = 3e8

    
    pass


#m=para()
#print(m.carrier_freq)