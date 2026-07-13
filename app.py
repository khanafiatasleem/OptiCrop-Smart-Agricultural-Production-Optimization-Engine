from flask import Flask, render_template, request
import joblib
import numpy as np
from crop_data import crop_data

app = Flask(__name__)

# Load trained model
model = joblib.load("model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/findcrop")
def findcrop():
    return render_template("findcrop.html")


@app.route("/predict", methods=["POST"])
def predict():

    # Read form values
    N = float(request.form["N"])
    P = float(request.form["P"])
    K = float(request.form["K"])
    temperature = float(request.form["temperature"])
    humidity = float(request.form["humidity"])
    ph = float(request.form["ph"])
    rainfall = float(request.form["rainfall"])

    # Same preprocessing as training
    K = np.log1p(K)

    sample = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

    # Predict crop
    prediction = model.predict(sample)[0]

    # Prediction confidence (if supported)
    confidence = None
    if hasattr(model, "predict_proba"):
        confidence = round(max(model.predict_proba(sample)[0]) * 100, 2)

    # Get crop details
    info = crop_data.get(
        prediction.lower(),
        {
            "image": "default.jpg",
            "season": "Not Available",
            "temperature": "-",
            "rainfall": "-",
            "ph": "-",
            "water": "-",
            "yield": "-",
            "tips": [
                "Follow soil testing recommendations.",
                "Maintain proper irrigation.",
                "Use balanced fertilizers."
            ]
        }
    )

    return render_template(
        "result.html",
        prediction=prediction,
        confidence=confidence,
        info=info
    )


if __name__ == "__main__":
    app.run(debug=True)