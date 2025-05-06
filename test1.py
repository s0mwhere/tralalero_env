import numpy as np
a=1
beamforming_matrix = np.array(((1+5j,2+6j,3+3j),(4+1j,5+3j,6+5j)))

cal_beam_power = np.trace(np.matmul((beamforming_matrix.conj().T),(beamforming_matrix)))

print(cal_beam_power<a)