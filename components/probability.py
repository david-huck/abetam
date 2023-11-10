import numpy as np

def beta_mode_from_params(a,b):
    mode = (a-1)/(a+b-2)
    return mode

def beta_dist_params_from_mode(mode, base_val=8):
    # mode = (a-1)/(a+b-2)
    if mode > 0.5:
        a = base_val
        b = ((a - 1 - mode * a) + 2 * mode) / mode
    else:
        b = base_val
        a = (mode * (b - 2) + 1) / (1 - mode)
    return a, b


def beta_with_mode_at(mode, n, interval=(-1, 1)):
    assert interval[0] < interval[1], ValueError(
        "Intervals must be specified as (x,y) where x<y!"
    )

    a, b = beta_dist_params_from_mode(mode)
    rand_vals = np.random.beta(a, b, n)
    # stretch to fit interval
    if interval != (0, 1):
        int_len = interval[1] - interval[0]
        rand_vals = rand_vals * int_len + interval[0]
    return rand_vals