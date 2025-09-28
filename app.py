from flask import Flask,render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open('random_forest_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the scaler
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    area = float(request.form['area'])
    bathrooms = float(request.form['bathrooms'])
    bhk = float(request.form['bhk'])
    house_type = request.form['type']
    furnishing = request.form['furnishing']
    parking = int(request.form["parking"])
    # Apply logarithmic transformation
    area = np.log(area + 1)
    bathrooms = np.log(bathrooms + 1)
    bhk = np.log(bhk + 1)
    parking = np.log(parking +1)
    # Create dummy variables
    type_apartment = 1 if house_type == 'Apartment' else 0
    type_building = 1 if house_type == 'Building' else 0

    furnishing_furnished = 1 if furnishing == 'Furnished' else 0
    furnishing_semi_furnished = 1 if furnishing == 'Semi-Furnished' else 0
    furnishing_unfurnished = 1 if furnishing == 'Unfurnished' else 0
    
    # Combine all features into a single array
    # features = np.array([area, bathrooms, bhk, furnishing_furnished, furnishing_semi_furnished, type_apartment, type_building]).reshape(1, -1)
    features = np.array([area, bhk, bathrooms,parking, furnishing_furnished, furnishing_semi_furnished,furnishing_unfurnished, type_apartment, type_building]).reshape(1, -1)
    
    # Scale the features
    features_scaled = scaler.transform(features)
    
    # Predict the price
    prediction = model.predict(features_scaled)
    
    return "Rs "+ str(int(prediction[0]))

if __name__ == "__main__":
    app.run(debug=True)

