""" Test script for SPM functions

Run these tests with::

    python3 code/tests/test_spm_funcs.py

or better, in IPython::

    %run code/tests/test_spm_funcs.py
"""

from os.path import dirname, abspath, join as pjoin
import sys

MY_DIR = dirname(__file__)
EXAMPLE_FILENAME = 'ds107_sub012_t1r2_small.nii'

# Here you should add the code directory to the Python path
# LAB(begin solution)
CODE_DIR = abspath(pjoin(MY_DIR, '..'))
sys.path.append(CODE_DIR)
# LAB(replace solution)
# +++your code here+++
# LAB(end solution)

import numpy as np

import nibabel as nib

# This import needs the code directory on the Python PATH
from spm_funcs import get_spm_globals, spm_global


def test_spm_globals():
    # Test get_spm_globals and spm_global functions
    example_path = pjoin(MY_DIR, EXAMPLE_FILENAME)
    expected_values = np.loadtxt(pjoin(MY_DIR, 'global_signals.txt'))
    glob_vals = get_spm_globals(example_path)
    assert np.allclose(glob_vals, expected_values, rtol=1e-4)
    img = nib.load(example_path)
    data = img.get_data()
    globals = []
    for vol_no in range(data.shape[-1]):
        vol = data[..., vol_no]
        globals.append(spm_global(vol))
    assert np.allclose(globals, expected_values, rtol=1e-4)


if __name__ == '__main__':
    # File being executed as a script
    test_spm_globals()
