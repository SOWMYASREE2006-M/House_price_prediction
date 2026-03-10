import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Load dataset
data = pd.read_csv("data/house_data.csv")

# Features
X = data[['area','bedrooms','bathrooms','floors','age','parking']]

# Target
y = data['price']

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained successfully")