from flask import Flask, request, jsonify, render_template
from pymongo import MongoClient
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder  
from xgboost import XGBRegressor
import pickle

app = Flask(__name__)

# MongoDB connection
client = MongoClient("mongodb://localhost:27017/")  # Replace with your connection string
db = client["disaster_management"]  # Replace with your database name

# Collections
collection_1 = db["collection_1"]
collection_2 = db["collection_2"]
collection_3 = db["collection_3"]

# Load the pre-trained XGBoost model
with open("xgb_model.pkl", "rb") as f:
    model_data = pickle.load(f)
xgb_model = model_data["model"]


print(type(xgb_model))
# Preprocessing configuration
categorical_columns = ['Country', 'Disaster Group', 'Disaster Type', 'Disaster Subtype']
numerical_columns = ['Year', 'Total Events', 'Total Affected', 'Total Deaths', 'CPI']
encoder = OneHotEncoder(drop='first', sparse=False)

# Assume `encoder` was fitted during training, you can save/load the encoder using pickle as well.

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/crud")
def crud():
    return render_template("crud.html")

@app.route("/predict")
def predict():
    return render_template("predict.html")

import logging
logging.basicConfig(level=logging.DEBUG)

@app.route("/predict-result", methods=["POST"])
def predict_disaster_damage():
    try:
        # Collect inputs from the frontend
        data = request.json
        logging.debug(f"Received data: {data}")
        
        # Extract and preprocess inputs
        year = int(data.get('Year'))
        total_events = int(data.get('Total Events'))
        total_affected = int(data.get('Total Affected'))
        total_deaths = int(data.get('Total Deaths'))
        cpi = float(data.get('CPI (Consumer Price Index)'))
        country = data.get('Country')
        disaster_group = data.get('Disaster Group')
        disaster_type = data.get('Disaster Type')
        disaster_subtype = data.get('Disaster Subtype')

        # Preprocess categorical features
        categorical_features = [country, disaster_group, disaster_type, disaster_subtype]
        logging.debug(f"Categorical features: {categorical_features}")

        # Load the encoder
        with open('encoder.pkl', 'rb') as f:
            encoder = pickle.load(f)

        encoded_features = encoder.transform([categorical_features])
        logging.debug(f"Encoded features: {encoded_features}")

        # Combine numerical and encoded categorical features
        input_features = np.concatenate((
            np.array([year, total_events, total_affected, total_deaths, cpi]).reshape(1, -1),
            encoded_features
        ), axis=1)
        logging.debug(f"Input features for prediction: {input_features}")

        # Make a prediction
        log_prediction = xgb_model.predict(input_features)[0]
        prediction = np.expm1(log_prediction)  # Convert log-transformed prediction back to original scale
        logging.debug(f"Log prediction: {log_prediction}, Final prediction: {prediction}")

        # Ensure prediction is a float
        prediction = float(prediction)

        # Return the prediction
        return jsonify({'prediction': prediction})

    except Exception as e:
        logging.exception("Error during prediction")
        return jsonify({'error': str(e)}), 500


@app.route("/insert-or-update", methods=["POST"])
def insert_or_update_data():
    content = request.json
    if not content:
        return jsonify({"error": "No data provided"}), 400

    record_id = content.get("Disaster_ID")  # Assuming `Disaster_ID` is the unique identifier
    if not record_id:
        return jsonify({"error": "No Disaster_ID provided"}), 400

    # Check if the record exists
    existing_record = collection_3.find_one({"Disaster_ID": record_id})
    if existing_record:
        # Update the existing record
        collection_3.update_one({"Disaster_ID": record_id}, {"$set": content})
        return jsonify({"message": "Data updated successfully!"}), 200
    else:
        # Insert new data
        result = collection_3.insert_one(content)
        return jsonify({"message": "Data inserted successfully!", "id": str(result.inserted_id)}), 201


@app.route("/get", methods=["GET"])
def get_data():
    data = list(collection_3.find({}, {"_id": 0}))  # Exclude `_id` for simplicity
    return jsonify(data), 200


@app.route("/delete/<record_id>", methods=["DELETE"])
def delete_data(record_id):
    result = collection_3.delete_one({"Disaster_ID": int(record_id)})  # Assuming `record_id` is numeric
    if result.deleted_count:
        return jsonify({"message": "Record deleted successfully!"}), 200
    return jsonify({"error": "Record not found"}), 404


if __name__ == "__main__":
    app.run(debug=True)


