from math import sqrt

import numpy as np
from scipy import fftpack as fp

from ..util_math import dft2, idft2


# --- high pass filters ---


def high_pass_ideal(d, cutoff):
    if d >= cutoff:
        return 1
    else:
        return 0


def high_pass_gauss(d, cutoff):
    return 1 - np.exp(-(d ** 2) / (2 * cutoff ** 2))


def high_pass_butterworth(d, cutoff, order):
    if d == 0:
        return 0
    return 1.0 / (1 + (cutoff / d) ** (2 * order))


# --- low pass filters ---


def low_pass_ideal(d, cutoff):
    if d <= cutoff:
        return 1
    else:
        return 0


def low_pass_gauss(d, cutoff):
    return np.exp(-(d ** 2) / (2 * cutoff ** 2))


def low_pass_butterworth(d, cutoff, order):
    return 1.0 / (1 + (d / cutoff) ** (2 * order))


# --- band pass filters ---


def band_pass_ideal(d, cutoff, width):
    if cutoff - width / 2.0 <= d <= cutoff + width / 2.0:
        return 1
    else:
        return 0


def band_pass_gauss(d, cutoff, width):
    if d == 0:
        return 0
    return np.exp(-((d ** 2 - cutoff ** 2)/(d * width)) ** 2)


def band_pass_butterworth(d, cutoff, width, order):
    if d == 0:
        return 0
    return 1 / (1 + ((d ** 2 - cutoff ** 2) / (d * width)) ** (2 * order))


# --- band reject filters ---


def band_reject_ideal(d, cutoff, width):
    if cutoff - width / 2.0 <= d <= cutoff + width / 2.0:
        return 0
    else:
        return 1


def band_reject_gauss(d, cutoff, width):
    if d == 0:
        return 1
    return 1 - np.exp(-((d ** 2 - cutoff ** 2)/(d * width)) ** 2)


def band_reject_butterworth(d, cutoff, width, order):
    if d ** 2 - cutoff ** 2 == 0:
        return 1
    return 1 / (1 + ((d * width) / (d ** 2 - cutoff ** 2)) ** (2 * order))


def distance(u, v, (M, N)):
    return sqrt((u - M / 2.0) ** 2 + (v - N / 2.0) ** 2)


def filter_image(image, filter, options):
    im_dft = fp.fftshift(dft2(image))

    result = np.zeros(im_dft.shape, dtype=np.complex)
    for u in range(im_dft.shape[0]):
        for v in range(im_dft.shape[1]):
            d = distance(u, v, im_dft.shape)
            result[u, v] = filter(d, **options) * im_dft[u, v]

    return abs(idft2(result))


LOW_PASS_FILTERS = {
    'ideal': low_pass_ideal,
    'gauss': low_pass_gauss,
    'butterworth': low_pass_butterworth,
}

HIGH_PASS_FILTERS = {
    'ideal': high_pass_ideal,
    'gauss': high_pass_gauss,
    'butterworth': high_pass_butterworth,
}

BAND_PASS_FILTERS = {
    'ideal': band_pass_ideal,
    'gauss': band_pass_gauss,
    'butterworth': band_pass_butterworth,
}

BAND_REJECT_FILTERS = {
    'ideal': band_reject_ideal,
    'gauss': band_reject_gauss,
    'butterworth': band_reject_butterworth,
}


FILTERS = {
    'low_pass': LOW_PASS_FILTERS,
    'high_pass': HIGH_PASS_FILTERS,
    'band_pass': BAND_PASS_FILTERS,
    'band_reject': BAND_REJECT_FILTERS,
}
