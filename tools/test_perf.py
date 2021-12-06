#!/usr/bin/env python3

"""This script checks and outputs to STDOUT whether performance optimizations
are enabled.
"""

import os
from tensorflow.python.util import _pywrap_util_port


def get_mkl_enabled_flag():
    """Checks whether MKL is enabled. This is off of Intel's sanity check.

    Copyright Intel 2021
    """
    mkl_enabled = False

    onednn_enabled = int(os.environ.get('TF_ENABLE_ONEDNN_OPTS', '0'))
    mkl_enabled = _pywrap_util_port.IsMklEnabled() or (onednn_enabled == 1)

    return mkl_enabled


if __name__ == '__main__':
    print(f'MKL Enabled: {get_mkl_enabled_flag()}')
