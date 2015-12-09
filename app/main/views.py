import os

import numpy as np

from scipy import ndimage
from scipy import fftpack as fp
from matplotlib import pyplot, cm

from PIL import Image

from flask import Blueprint, render_template, url_for, send_file
from flask import redirect, request, g, current_app

from .util import get_image_path, get_no_image_path, clear_data_folder
from .util import get_image_url, dft2, idft2, showfft, check_image

main = Blueprint('main', __name__, template_folder='templates')


VALID_EXTENSIONS = {'png', 'jpg'}

IMAGE_TYPES = {'origin', 'fft', 'fft-real', 'fft-imag', 'rec', 'ref'}


@main.route('/')
def index():
    return redirect(url_for('main.upload'))


@main.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files['file']
        image = Image.open(file)
        if file:
            _, ext = os.path.splitext(file.filename)
            ext = ext[1:]
            if ext in VALID_EXTENSIONS:
                clear_data_folder()

                response = current_app.make_response(redirect(url_for('.upload')))
                g.current_image = ext
                response.set_cookie('current_image', g.current_image)

                image.convert('L').save(get_image_path())
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

        pyplot.imshow(showfft(abs(dft_res)), cmap=cm.Greys_r)
        pyplot.savefig(get_image_path(type='fft'))

        pyplot.imshow(showfft(np.real(dft_res)), cmap=cm.Greys_r)
        pyplot.savefig(get_image_path(type='fft-real'))

        pyplot.imshow(showfft(np.imag(dft_res)), cmap=cm.Greys_r)
        pyplot.savefig(get_image_path(type='fft-imag'))

        return redirect(url_for('.fourier'))
    return render_template('main/fourier.jinja')


@main.route('/inv-fourier', methods=['GET', 'POST'])
@check_image
def inv_fourier():
    if request.method == 'POST':
        image = ndimage.imread(get_image_path())
        rec_image = idft2(dft2(image))

        pyplot.imshow(image, cmap=cm.Greys_r)
        pyplot.savefig(get_image_path(type='ref'))

        pyplot.imshow(abs(rec_image), cmap=cm.Greys_r)
        pyplot.savefig(get_image_path(type='rec'))

        return redirect(url_for('.inv_fourier'))
    return render_template('main/inv_fourier.jinja')


@main.route('/about')
def about():
    return render_template('main/about.jinja')
