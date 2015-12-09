import os

import numpy as np

from scipy import ndimage
from scipy import fftpack as fp
from matplotlib import pyplot

from flask import Blueprint, render_template, url_for, send_file
from flask import redirect, request, g, current_app

from .util import get_image_path, get_no_image_path, clear_data_folder
from .util import get_image_url, dft2, showfft, check_image

main = Blueprint('main', __name__, template_folder='templates')


VALID_EXTENSIONS = {'png', 'jpg'}

IMAGE_TYPES = {'origin', 'fft', 'fft-real', 'fft-imag'}


@main.route('/')
def index():
    return redirect(url_for('main.upload'))


@main.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        image = request.files['file']
        if image:
            _, ext = os.path.splitext(image.filename)
            ext = ext[1:]
            if ext in VALID_EXTENSIONS:
                clear_data_folder()

                response = current_app.make_response(redirect(url_for('.upload')))
                g.current_image = ext
                response.set_cookie('current_image', g.current_image)

                image.save(get_image_path())
                return response
        return redirect(url_for('.upload'))
    return render_template('main/upload.jinja', image_url=get_image_url())


@main.route('/image')
def get_image():
    type = request.args['type']

    if type not in IMAGE_TYPES or not g.get('current_image'):
        image_path = get_no_image_path()
    else:
        image_path = get_image_path(type=type)

    if not os.path.exists(image_path):
        image_path = get_no_image_path()

    return send_file(image_path, mimetype='image/gif')


@main.route('/fourier', methods=['GET', 'POST'])
@check_image
def fourier():
    if request.method == 'POST':
        image = ndimage.imread(get_image_path())
        dft_res = fp.fftshift(dft2(image))

        pyplot.imshow(showfft(abs(dft_res)))
        pyplot.savefig(get_image_path(type='fft'))

        pyplot.imshow(showfft(np.real(dft_res)))
        pyplot.savefig(get_image_path(type='fft-real'))

        pyplot.imshow(showfft(np.imag(dft_res)))
        pyplot.savefig(get_image_path(type='fft-imag'))

        return redirect(url_for('main.fourier'))
    return render_template('main/fourier.jinja')


@main.route('/about')
def about():
    return render_template('main/about.jinja')
