import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import pickle

# Load and preprocess the data
data = pd.read_csv("MagicBricks.csv").dropna()
print(data.info())
data["Area"] = np.log(data["Area"] + 1)
data["BHK"] = np.log(data["BHK"] + 1)
data["Bathroom"] = np.log(data["Bathroom"] + 1)
data["Parking"] = np.log(data["Parking"]+1)
# Create dummy variables for categorical features
data = pd.get_dummies(data, columns=["Furnishing", "Type"])

# Separate features and target variable
Prices = data["Price"]
data = data.drop(["Price", "Per_Sqft", "Locality", "Transaction", "Status"], axis=1)

# Split the data into training and testing sets
print(data.columns)
X_train, X_test, Y_train, Y_test = train_test_split(data, Prices, test_size=0.2, random_state=44)

# Scale the features
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
print(X_train.shape)
# Train the RandomForestRegressor model
forest = RandomForestRegressor(random_state=44)
forest.fit(X_train_s, Y_train)
print(forest.score(X_test_s, Y_test))

# Save the trained model
with open('random_forest_model.pkl', 'wb') as file:
    pickle.dump(forest, file)

# Save the scaler
with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)
