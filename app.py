from flask import Flask, render_template, request, jsonify
import joblib
import json
import numpy as np


app = Flask(__name__)


# Load
knn_model = joblib.load("knn_model.pkl")
knn_scaler = joblib.load("knn_scaler.pkl")
rf_model = joblib.load("random_forest_model.pkl")
with open("evaluation_results.json", "r") as file:
    evaluation_data = json.load(file)

# Deskripsi model
model_info = {
    "KNN": {
        "name": "K-Nearest Neighbors",
        "description": "Prediksi berdasarkan kedekatan data tetangga.",
        "metrics": evaluation_data["evaluation_metrics"]["KNN"],
        "parameters": evaluation_data["best_parameters"]["KNN"]
    },

    "RF": {
        "name": "Random Forest",
        "description": "Prediksi menggunakan kumpulan decision trees.",
        "metrics": evaluation_data["evaluation_metrics"]["RandomForest"],
        "parameters": evaluation_data["best_parameters"]["RandomForest"]
    }
}

# Fitur
FEATURES = ["AveRooms", "Latitude", "Longitude", "EstLandArea", "EstBuildArea"]


# Route untuk index.html
@app.route("/")
def home():
    return render_template(
        "index.html",
        model_info=model_info
    )


# API Prediksi
@app.route("/predict", methods=["POST"])
def predict():

    try:
        data = request.get_json()
        selected_model = data.get("model")
        if selected_model not in ["KNN", "RF"]:
            return jsonify({"error": "Model tidak valid."}), 400

        # Untuk menyusun order fitur
        features = []
        for feature in FEATURES:
            value = float(data[feature])
            if not np.isfinite(value):
                raise ValueError(f"Nilai {feature} tidak valid.")
            features.append(value)

        input_data = np.array(features).reshape(1, -1) # Konversi ke 2D array

        # Selection untuk model
        if selected_model == "KNN":
            input_scaled = knn_scaler.transform(input_data)
            prediction = knn_model.predict(input_scaled)[0]
        else:
            prediction = rf_model.predict(input_data)[0]

        return jsonify({
            "prediction": float(prediction),
            "prediction_usd": float(prediction * 100000) # MedHouseVal menggunakan satuan $100,000 USD.
        })

    except (ValueError, TypeError, KeyError) as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": "Terjadi kesalahan saat melakukan prediksi: " + str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
