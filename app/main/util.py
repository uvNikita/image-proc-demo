import os

from flask import current_app


def get_image_path(name):
    return os.path.join(current_app.config['APP_DATA_FOLDER'],
                        '{}.png'.format(name))


def get_no_image_path():
    return os.path.join(current_app.static_folder, 'no-image.png')
