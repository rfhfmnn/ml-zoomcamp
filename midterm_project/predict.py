#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Machine Learning Zoomcamp — Midterm Project
Model Serving Web Application (predict.py)

This service serves predictions for Club Atlético Independiente match outcomes
over a REST API using Flask.
"""

import os
import pickle
from flask import Flask, request, jsonify

MODEL_FILE = os.environ.get("MODEL_PATH", "model.bin")
OPTIMAL_THRESHOLD = float(os.environ.get("OPTIMAL_THRESHOLD", 0.41))

# Initialize Flask Application
app = Flask("cai-match-predictor")

# Load model and vectorizer at startup
if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError(f"Model file not found at: {MODEL_FILE}")

with open(MODEL_FILE, "rb") as f_in:
    dv, model = pickle.load(f_in)

print(f"Loaded model and DictVectorizer from '{MODEL_FILE}'.")
print(f"Optimal decision threshold: {OPTIMAL_THRESHOLD}")


@app.route("/ping", methods=["GET"])
def ping():
    """Healthcheck endpoint."""
    return jsonify({"status": "healthy", "service": "cai-match-predictor"}), 200


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict match outcome from JSON payload.

    Expected JSON Schema:
    {
        "rival": "Racing Club",
        "condicion_independiente": "Local",  # or "Visitante"
        "anio": 2026
    }
    """
    match_data = request.get_json(silent=True)
    if not match_data:
        return jsonify({"error": "Invalid or missing JSON body in request"}), 400

    # Validate required fields
    required_fields = ["rival", "condicion_independiente"]
    missing = [f for f in required_fields if f not in match_data]
    if missing:
        return jsonify({"error": f"Missing required fields: {missing}"}), 400

    # Ensure defaults if optional fields are missing
    match_payload = {
        "rival": str(match_data.get("rival")),
        "condicion_independiente": str(match_data.get("condicion_independiente")),
        "anio": int(match_data.get("anio", 2026))
    }

    # Vectorize and predict
    X = dv.transform([match_payload])
    win_probability = float(model.predict_proba(X)[0, 1])
    is_victory = bool(win_probability >= OPTIMAL_THRESHOLD)

    response = {
        "match": match_payload,
        "win_probability": round(win_probability, 4),
        "non_win_probability": round(1.0 - win_probability, 4),
        "threshold_used": OPTIMAL_THRESHOLD,
        "victory": is_victory,
        "prediction": "Victoria" if is_victory else "No Victoria"
    }

    return jsonify(response), 200


if __name__ == "__main__":
    # Check if waitress is available for production-quality WSGI serving on Windows
    try:
        from waitress import serve
        print("Starting production WSGI server with Waitress on port 9696...")
        serve(app, host="0.0.0.0", port=9696)
    except ImportError:
        print("Starting Flask development server on port 9696...")
        app.run(debug=False, host="0.0.0.0", port=9696)
