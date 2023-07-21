import numpy as np

def noise_and_clip(data):
    '''
    Adding some noise to normalized dataset
    Arguments:
        data: normalized dataset
    Return:
        same normalized dataset, but with noise
    '''
    noise = np.random.normal(loc=0.0, scale=0.1, size=data.shape)
    return np.clip(data + noise, 0., 1.)