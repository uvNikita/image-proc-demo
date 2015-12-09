from math import sqrt
from collections import namedtuple

import numpy as np
from scipy import fftpack as fp

from .util import dft2, idft2


Filter = namedtuple('Filter', ['id', 'name', 'func'])


def high_pass(d, cutoff):
    if d >= cutoff:
        return 1
    else:
        return 0


def low_pass(d, cutoff):
    if d <= cutoff:
        return 1
    else:
        return 0


def distance(u, v, (M, N)):
    return sqrt((u - M / 2.0) ** 2 + (v - N / 2.0) ** 2)


def filter_image(image, filter, **options):
    im_dft = fp.fftshift(dft2(image))

    result = np.zeros(im_dft.shape, dtype=np.complex)
    for u in range(im_dft.shape[0]):
        for v in range(im_dft.shape[1]):
            d = distance(u, v, im_dft.shape)
            result[u, v] = filter.func(d, **options) * im_dft[u, v]

    return abs(idft2(result))


FILTERS = {
    'high_pass': Filter('high_pass', 'High Pass', high_pass),
    'low_pass': Filter('low_pass', 'Low Pass', low_pass),
}


