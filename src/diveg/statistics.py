import numpy as np


def iqr(a: np.ndarray) -> float:
    """
    Calculate the range covered by the middle 50 % of the input values.

    Using statistical terms, calculate the difference between the third
    and the first quartiles (the inter-quartile range, IQR).

    Example
    -------

    >>> x = np.arange(6) * 5
    # array([ 0,  5, 10, 15, 20])
    >>> np.cumsum(x)
    # array([ 0,  5, 15, 30, 50], dtype=int32)
    >>> a, b = np.percentile(x, [25, 75])
    >>> (a, b)
    # (5.0, 15.0)
    >>> IQR = b - a
    >>> IQR
    # 10.0

    """
    a, b = np.percentile(np.asarray(a), [25, 75])
    return b - a
