import numpy as np

def normalize(var: np.array, direction=1):
    """normalizes the values to the interval [0, 1].

    Args:
        `var` (np.array): The array to be normalised.
        `direction` (int, optional): indicates wether higher is better (1) or lower is better (-1). Defaults to 1.

    Returns:
        pd.Series: The normalised series.
    """
    # if direction:
    max_ = var.max()
    min_ = var.min()
    if direction == 1:
        norm = (var - min_) / (max_ - min_)
    elif direction == -1:
        norm = (max_ - var) / (max_ - min_)

    return norm
