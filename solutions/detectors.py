""" Utilities for detecting outliers

These functions take a vector of values, and return a boolean vector of the
same length as the input, where True indicates the corresponding value is an
outlier.
"""

# Python 2 compatibility
from __future__ import print_function, division

# Any imports you need
# LAB(begin solution)
import numpy as np
# LAB(replace solution)
# +++your code here+++
# LAB(end solution)


def iqr_detector(measures, iqr_proportion=1.5):
    """ Detect outliers in `measures` using interquartile range.

    Returns a boolean vector of same length as `measures`, where True means the
    corresponding value in `measures` is an outlier.

    Call Q1, Q2 and Q3 the 25th, 50th and 75th percentiles of `measures`.

    The interquartile range (IQR) is Q3 - Q1.

    An outlier is any value in `measures` that is either:

    * > Q3 + IQR * `iqr_proportion` or
    * < Q1 - IQR * `iqr_proportion`.

    See: https://en.wikipedia.org/wiki/Interquartile_range

    Parameters
    ----------
    measures : 1D array
        Values for which we will detect outliers
    iqr_proportion : float, optional
        Scalar to multiply the IQR to form upper and lower threshold (see
        above).  Default is 1.5.

    Returns
    -------
    outlier_tf : 1D boolean array
        A boolean vector of same length as `measures`, where True means the
        corresponding value in `measures` is an outlier.
    """
    # Any imports you need
    # LAB(begin solution)
    q1, q3 = np.percentile(measures, [25, 75])
    iqr = q3 - q1
    up_thresh = q3 + iqr * iqr_proportion
    down_thresh = q1 - iqr * iqr_proportion
    return np.logical_or(measures > up_thresh, measures < down_thresh)
    # LAB(replace solution)
    # Hints:
    # * investigate np.percentile
    # * You'll likely need np.logical_or
    # +++your code here+++
    # LAB(end solution)
