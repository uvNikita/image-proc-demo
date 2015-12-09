import os

from flask import Flask, g, request

from .main.views import main
from .main.util import ensure_data_folder


def create_app():
    app = Flask(
        __name__,
        static_folder=os.path.join(os.path.dirname(__file__), '..', 'static')
    )

    app.config.from_object('config')

    app.register_blueprint(main)

    @app.before_request
    def get_current_image():
        g.current_image = request.cookies.get('current_image')

    ensure_data_folder(app.config)

    return app
