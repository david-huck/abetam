import numpy as np


def beta_mode_from_params(a: float, b: float):
    """Calculate the mode of a beta distribution from the shape parameters a and b.

    # Parameters
    a : float
        First shape parameter of the beta distribution
    b : float
        Second shape parameter of the beta distribution 

    # Returns
    mode : float
        The mode of the beta distribution with shape parameters a and b.

    # Notes
    The mode of a beta distribution with shape parameters a and b is given by:
        mode = (a - 1) / (a + b - 2)

    # Examples
    >>> beta_mode_from_params(2, 5)
    0.2
    >>> beta_mode_from_params(5, 5)
    0.5
    >>> beta_mode_from_params(7, 5.)
    0.6
    """
    mode = (a - 1) / (a + b - 2)
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
    """Generate `n` random values from a beta distribution 
    with the given `mode` and interval.

    Args:
        mode (float): Mode of the beta distribution 
        n (int): Number of values to generate
        interval (tuple): Min and max values for output  

    Returns:
        np.ndarray: n random values from the distribution
    """
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
