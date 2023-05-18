'''
Helper functions to handle noise scheduling. This leaves room for multiple
different noise schedules to be added, if desired.

@Author: George Witt
'''

import numpy as np
import torch

def linear_var_schedule(b_start, b_end, n_steps):
    return np.linspace(b_start, b_end, n_steps)

def calc_alpha_ts_bar(alpha_ts):
    return np.cumprod(alpha_ts, axis=0)

def calc_alpha_ts(B_schedule):
    return 1. - B_schedule

def gauss_noise_step(images, alpha_ts_bar, t):
    '''
    @image: B x [28 x 28] torch matrices
    @alpha_ts_bar: numpy matrix of cumulative products for all time. 
    @t: integer timestep for the noise schedule

    NO VALUES above can be None.
    '''

    # Note this method employs the step forward trick
    # It was noted in (https://arxiv.org/pdf/2006.11239.pdf), Ho et al. 
    # that it is possible to effectively 'jump' through sampling to t.
    # So even though this generates samples for t, it skips the step process
    # and applies the noise all in one go.
    # This is from equation (4) in Ho et al.

    #alpha_ts = 1. - B # [vector by t]
    #alpha_ts_bar = np.cumprod(alpha_ts, axis=0) # [vector by t] of cumulative products.

    sqrt_alpha_ts_bar = np.sqrt(alpha_ts_bar[t])
    sqrt_one_minus_atsbar = np.sqrt(1 - alpha_ts_bar[t])  

    # Now we can calculate the random uniform noise ~ N(0, I)
    noise = torch.randn_like(images)

    # see alg 1.
    ret = sqrt_alpha_ts_bar * images + sqrt_one_minus_atsbar * noise, noise

    return ret


