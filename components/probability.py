import numpy as np
from scipy.optimize import minimize


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


def beta_with_mode_at(mode, n, interval=(0, 1)):
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
    rand_vals = np.float32(np.random.beta(a, b, n))
    # stretch to fit interval
    if interval != (0, 1):
        int_len = interval[1] - interval[0]
        rand_vals = rand_vals * int_len + interval[0]
    return rand_vals


def normal_truncated(mean, std, size, trunc_interval=(0, 1)):
    dist = np.random.normal(loc=mean, scale=std, size=size)

    if trunc_interval:
        lower, upper = trunc_interval
        dist[dist < lower] = lower  # -dist[dist<lower]
        dist[dist > upper] = upper  # upper - (dist[dist>upper]-upper)

    return dist


def desired_modes_from_price_mode(price_mode, rel_att_weight=2, rel_em_weight=1):
    mode_aw = (1 - price_mode) * rel_att_weight / 3
    mode_ew = (1 - price_mode) * rel_em_weight / 3
    return [price_mode, mode_aw, mode_ew]


def dirichlet_modes(alpha):
    alpha = np.array(alpha)
    return (alpha - 1) / (sum(alpha) - len(alpha))


def dirichlet_alphas(modes):
    # define function to minimize deviation
    func = lambda x: np.linalg.norm(modes - dirichlet_modes(x))

    # minimize (apparently sensitive to x0)
    res = minimize(func, x0=[5, 4, 3], method="L-BFGS-B")

    # assert success and a small maximal difference
    max_diff = np.max(np.abs((dirichlet_modes(res.x) - modes)))
    assert max_diff < 1e-6, AssertionError(f"{max_diff=}")
    assert res.success, AssertionError(f"{res=}")
    return res.x
