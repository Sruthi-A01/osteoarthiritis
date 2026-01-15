from flask import Blueprint, render_template, request
import os
import pandas as pd
from werkzeug.utils import secure_filename

from app.config import Config
from app.services.feature_engineering import engineer_features
from app.services.training import prepare_and_train
from app.services.visualization import clear_plots, save_confusion, save_roc
from app.models import MODELS

routes = Blueprint("routes", __name__)


@routes.route("/", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        clear_plots()

        if "file" not in request.files or request.files["file"].filename == "":
            return render_template("upload.html", error="Please upload a file")

        file = request.files["file"]
        file_path = os.path.join(
            Config.UPLOAD_FOLDER,
            secure_filename(file.filename)
        )
        file.save(file_path)

        # Read Excel
        df_raw = pd.read_excel(file_path)

        # Feature engineering
        df_model = engineer_features(df_raw)

        # Train models
        fitted_models, X_test, y_test = prepare_and_train(
            df_model, MODELS, Config.RANDOM_STATE
        )

        plots = []

        for name, model in fitted_models.items():
            y_proba = model.predict_proba(X_test)[:, 1]
            y_pred = (y_proba >= Config.OPTIMAL_THRESHOLD).astype(int)

            plots.append(save_confusion(y_test, y_pred, name))
            plots.append(save_roc(y_test, y_proba, name))

        return render_template("dashboard.html", plots=plots)

    return render_template("upload.html")
