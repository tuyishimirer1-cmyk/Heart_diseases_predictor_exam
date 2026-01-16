import os
import joblib
from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np

base_dir = os.path.dirname(os.path.abspath(__file__))
artifacts_dir = os.path.join(base_dir, "artifacts")

model = joblib.load(os.path.join(artifacts_dir, "best_model.pkl"))

with open(os.path.join(artifacts_dir, "feature_columns.txt")) as f:
    feature_columns = [line.strip() for line in f]

with open(os.path.join(artifacts_dir, "class_names.txt")) as f:
    class_names = [line.strip() for line in f]

app = Flask(__name__, template_folder="templates", static_folder="static")

@app.route("/")
def home():
    return render_template("index_25rp20399.html", class_names=class_names)

@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        input_df = pd.DataFrame([data])
        input_df = input_df.reindex(columns=feature_columns)
        input_df = input_df.apply(pd.to_numeric, errors='coerce')

        if input_df.isnull().any().any():
            return jsonify({"error": "Invalid or missing input. Ensure all fields are numeric."}), 400

        probs = model.predict_proba(input_df)[0]
        pred_class = int(np.argmax(probs))

        response = {
            "predicted_class": class_names[pred_class],
            "confidence": float(probs[pred_class]),
            "probabilities": {class_names[i]: float(probs[i]) for i in range(len(probs))}
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
