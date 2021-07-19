"""
@author:    Guan'an Wang
@contact:   guan.wang0706@gmail.com
"""


import numpy as np


def hamming_distance(x, y):
    """
    compute hamming distance (NOT hamming similarity)
    between x and y
    Args:
        x(np.ndarray): [num_x, dim] in {0, 1}
        y(np.ndarray): [num_y, dim] in {0, 1}
    Return:
        (np.ndarray): [num_x, num_y]
    """
    assert min(x.min(), y.min())==0 and max(x.max(), y.max())==1, \
        'expect binary codes in \{0, 1\}, but got {{}, {}}'.format(min(x.min(), y.min()), max(x.max(), y.max()))

    assert x.shape[1] == y.shape[1], \
        'expect x and y have the same dimmension, but got x {} and y {}'.format(x.shape[1], y.shape[1])
    code_len = x.shape[1]

    x = (x-0.5)*2
    y = (y-0.5)*2
    return code_len - (np.matmul(x, y.transpose([1,0])) + code_len) / 2

