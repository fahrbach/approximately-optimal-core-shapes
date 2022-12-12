import numpy as np

def get_num_tucker_params(tensor, core_shape):
    num_tucker_params = np.prod(core_shape)
    for n in range(len(core_shape)):
        num_tucker_params += tensor.shape[n] * core_shape[n]
    return num_tucker_params
