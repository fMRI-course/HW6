""" Test script for detector functions

Run these tests with::

    python3 code/tests/test_detectors.py

or better, in IPython::

    %run code/tests/test_detectors.py
"""

from os.path import dirname, abspath, join as pjoin
import sys

MY_DIR = dirname(__file__)

# Here you should add the code directory to the Python path
# LAB(begin solution)
CODE_DIR = abspath(pjoin(MY_DIR, '..'))
sys.path.append(CODE_DIR)
# LAB(replace solution)
# +++your code here+++
# LAB(end solution)

import numpy as np

# This import needs the code directory on the Python PATH
from detectors import iqr_detector


def test_iqr_detector():
    # From: http://www.purplemath.com/modules/boxwhisk3.htm
    example_values = np.array(
        [10.2, 14.1, 14.4, 14.4, 14.4, 14.5, 14.5, 14.6, 14.7, 14.7, 14.7,
         14.9, 15.1, 15.9, 16.4])
    is_outlier = iqr_detector(example_values, 1.5)
    assert np.all(example_values[is_outlier] == [10.2, 15.9, 16.4])
    # Test not-default value for outlier proportion
    is_outlier = iqr_detector(example_values, 0.5)
    assert np.all(example_values[is_outlier] == [10.2, 14.1, 15.1, 15.9, 16.4])


if __name__ == '__main__':
    # File being executed as a script
    test_iqr_detector()
