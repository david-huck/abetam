import pandas as pd


def calc_score(row: pd.Series, weights: dict) -> float:
    """calculates the (mcda) score of the `row` using the `weights` by vector
    multiplication. The index of `row` (or the DataFrame on which `.apply()` is used)
    and the keys of the `weights` must match.
    """
    weight_series = pd.Series(weights)

    # ensure each index appears in both frames
    assert all(idx in weight_series.index for idx in row.index)
    # sort both series to ensure alignment
    row.sort_index(inplace=True)
    weight_series.sort_index(inplace=True)

    return row.values @ weight_series.values


def normalize(var: pd.Series, direction=1):
    """normalizes the values to the interval [0, 1].

    Args:
        `var` (pd.Series): The series to be normalised.
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
