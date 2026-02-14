from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("model/churn_model.pkl", "rb"))

@app.route("/")
def home():
    return "Customer Churn Prediction API is Running"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        gender = float(request.form["gender"])
        senior = float(request.form["senior"])
        partner = float(request.form["partner"])
        dependents = float(request.form["dependents"])
        tenure = float(request.form["tenure"])
        monthly = float(request.form["monthly"])
        total = float(request.form["total"])

        features = np.array([[gender, senior, partner, dependents, tenure, monthly, total]])

        prediction = model.predict(features)[0]

        return f"Prediction Result: {int(prediction)}"

    except Exception as e:
        return str(e)

if __name__ == "__main__":
    app.run(debug=True)
