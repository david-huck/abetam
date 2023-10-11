import pandas as pd


def calc_score(row: pd.Series, weights: dict) -> float:
    """calculates the (mcda) score of the `row` using the `weights` by vector
    multiplication. The index of `row` (or the DataFrame on which `.apply()` is used)
    and the keys of the `weights` must match.
    """
    weight_df = pd.DataFrame(
        weights.values(), columns=["weights"], index=weights.keys()
    )

    # ensure inputs are of equal length
    assert len(row) == len(weight_df)

    # ensure each index appears in both frames
    assert all(idx in weight_df.index for idx in row.index)
    return row @ weight_df["weights"]


def normalize(var: pd.Series):
    """normalizes the values to the interval [0, 1]."""
    max_ = var.max()
    min_ = var.min()
    norm = (var - min_) / (max_ - min_)
    return norm
