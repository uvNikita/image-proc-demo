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
from .filters import filter_image, FILTERS

main = Blueprint('main', __name__, template_folder='templates')


VALID_EXTENSIONS = {'png', 'jpg'}

IMAGE_TYPES = {
    'origin', 'fft', 'fft-real', 'fft-imag',
    'rec', 'ref',
    'filtered_low_pass', 'filtered_high_pass', 'filtered_band_pass',
}


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

                image = image.convert('L')
                image.save(get_image_path())

                pyplot.imshow(image, cmap=cm.Greys_r)
                pyplot.savefig(get_image_path(type='ref'))

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

        x_max = dft_res.shape[0]/2
        y_max = dft_res.shape[1]/2
        dims = [-x_max, x_max, -y_max, y_max]

        pyplot.imshow(showfft(abs(dft_res)), cmap=cm.Greys_r, extent=dims)
        pyplot.savefig(get_image_path(type='fft'))
        pyplot.close()

        pyplot.imshow(showfft(np.real(dft_res)), cmap=cm.Greys_r, extent=dims)
        pyplot.savefig(get_image_path(type='fft-real'))
        pyplot.close()

        pyplot.imshow(showfft(np.imag(dft_res)), cmap=cm.Greys_r, extent=dims)
        pyplot.savefig(get_image_path(type='fft-imag'))
        pyplot.close()

        return redirect(url_for('.fourier'))
    return render_template('main/fourier.jinja')


@main.route('/inv-fourier', methods=['GET', 'POST'])
@check_image
def inv_fourier():
    if request.method == 'POST':
        image = ndimage.imread(get_image_path())
        rec_image = idft2(dft2(image))

        pyplot.imshow(abs(rec_image), cmap=cm.Greys_r)
        pyplot.savefig(get_image_path(type='rec'))
        pyplot.close()

        return redirect(url_for('.inv_fourier'))
    return render_template('main/inv_fourier.jinja')


@main.route('/low-pass', methods=['GET', 'POST'])
@check_image
def low_pass():
    return process_filter('low_pass', ['cutoff'])


@main.route('/high-pass', methods=['GET', 'POST'])
@check_image
def high_pass():
    return process_filter('high_pass', ['cutoff'])


@main.route('/band-pass', methods=['GET', 'POST'])
@check_image
def band_pass():
    return process_filter('band_pass', ['cutoff', 'width'])


def process_filter(filter_type, options):
    if request.method == 'POST':
        filter_name = request.form['filter_name']
        print '*' * 100
        option_values = {
            option: float(request.form[option])
            for option in options
        }
        if filter_name == 'butterworth':
            option_values['order'] = int(request.form['order'])

        image = ndimage.imread(get_image_path())

        filter_func = FILTERS[filter_type][filter_name]
        pyplot.imshow(filter_image(image, filter_func, option_values), cmap=cm.Greys_r)
        pyplot.savefig(get_image_path(type='filtered_{}'.format(filter_type)))
        return redirect(url_for('.{}'.format(filter_type), **request.form))

    return render_template('main/filter.jinja', filter_type=filter_type, options=options)


@main.route('/about')
def about():
    return render_template('main/about.jinja')
