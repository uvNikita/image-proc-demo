import os

from flask import Flask

from .main.views import main


def create_app():
    app = Flask(
        __name__,
        static_folder=os.path.join(os.path.dirname(__file__), '..', 'static')
    )

    app.config.from_object('config')

    app.register_blueprint(main)

    if not os.path.exists(app.config['APP_DATA_FOLDER']):
        os.makedirs(app.config['APP_DATA_FOLDER'])

    return app
