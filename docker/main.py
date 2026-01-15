from flask import Flask
from app.config import Config
from app.routes import routes
import os


def create_app():
    app = Flask(
        __name__,
        template_folder="app/templates",
        static_folder="app/static"
    )

    app.config.from_object(Config)

    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    os.makedirs(app.config["PLOTS_FOLDER"], exist_ok=True)

    app.register_blueprint(routes)
    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=6055, debug=True)
