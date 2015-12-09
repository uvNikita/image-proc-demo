import os
import uuid
import shutil

from functools import wraps

import numpy as np

from scipy import fftpack as fp

from flask import current_app, url_for, g, redirect


def get_image_path(name='image', type='origin'):
    if not g.get('current_image'):
        raise Exception('No current image!')
    return os.path.join(current_app.config['APP_DATA_FOLDER'],
                        '{}-{}.{}'.format(name, type, g.current_image))


def get_image_url(name='image', type='origin'):
    return url_for('.get_image', name=name, type=type, random=uuid.uuid4())


def get_no_image_path():
    return os.path.join(current_app.static_folder, 'no-image.png')


def dft2(im):
    return fp.fft(fp.fft(im, axis=0), axis=1)


def showfft(fft):
    return np.log(1 + fft)


def check_image(func):
    @wraps(func)
    def inner(*args, **kwargs):
        if not g.get('current_image'):
            return redirect(url_for('.upload'))
        else:
            return func(*args, **kwargs)
    return inner


def clear_data_folder():
    shutil.rmtree(current_app.config['APP_DATA_FOLDER'])
    ensure_data_folder(current_app.config)


def ensure_data_folder(config):
    if not os.path.exists(config['APP_DATA_FOLDER']):
        os.makedirs(config['APP_DATA_FOLDER'])
