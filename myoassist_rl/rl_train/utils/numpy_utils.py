import numpy as np
def numpy2array(nparray:np.ndarray):
    return [int(x) if x.is_integer() else float(x) for x in nparray]