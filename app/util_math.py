import numpy as np
from scipy import fftpack as fp


def dft2(im):
    return fp.fft(fp.fft(im, axis=0), axis=1)


def idft2(im):
    return fp.ifft(fp.ifft(im, axis=0), axis=1)


def dct2(im):
    return fp.dct(fp.dct(im, norm='ortho', axis=0), norm='ortho', axis=1)


def idct2(im):
    return fp.idct(fp.idct(im, norm='ortho', axis=0), norm='ortho', axis=1)


def showfft(fft):
    return np.log(1 + fft)


def image_diff(im1, im2):
    # Mean Squared Error
    return np.sum((im1 - im2) ** 2) / (im1.shape[0] * im1.shape[1])
