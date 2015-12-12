from functools import partial

import numpy as np

from .util_math import dct2, dft2, idct2, idft2


def blockproc(im, fun, block_size=8):
    N = block_size
    new = np.zeros(im.shape, dtype='complex_')
    for i in range(im.shape[0] // N):
        for j in range(im.shape[1] // N):
            sl = slice(i*N, i*N + N), slice(j*N, j*N + N)
            new[sl] = fun(im[sl])
    return new


def compress_image(im, level, tr, itr):
    n = int(round(im.shape[0] * im.shape[1] * (1 - level)))

    imt = blockproc(im, tr)
    new_imt = np.zeros(imt.shape, dtype='complex_')

    I = abs(imt.ravel()).argsort()[-n:]
    I = [np.unravel_index(i, imt.shape) for i in I]
    for i in I:
        new_imt[i] = imt[i]
    return np.real(blockproc(new_imt, itr, block_size=8))

compress_dct = partial(compress_image, tr=dct2, itr=idct2)
compress_dft = partial(compress_image, tr=dft2, itr=idft2)