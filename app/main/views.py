import os
import uuid

from flask import Blueprint, render_template, url_for, send_file
from flask import redirect, request

from .util import get_image_path, get_no_image_path

main = Blueprint('main', __name__, template_folder='templates')


@main.route('/')
def index():
    return redirect(url_for('main.upload'))


@main.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        image = request.files['file']
        if image and image.filename.endswith('.png'):
            image.save(get_image_path('image'))
        return redirect(url_for('main.upload'))
    image_url = url_for('main.get_image', random=uuid.uuid4())
    return render_template('main/upload.jinja', image_url=image_url)


@main.route('/image')
def get_image():
    image_path = get_image_path('image')

    if not os.path.exists(image_path):
        image_path = get_no_image_path()

    return send_file(image_path, mimetype='image/gif')


@main.route('/about')
def about():
    return render_template('main/about.jinja')
