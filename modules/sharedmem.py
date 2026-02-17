import numpy as np

def init(shared_base, shared_shape):
    global shared_data
    shared_data = np.ctypeslib.as_array(shared_base.get_obj())
    shared_data = shared_data.reshape(shared_shape)