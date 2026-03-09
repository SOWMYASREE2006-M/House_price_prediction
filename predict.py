import joblib

model = joblib.load("model.pkl")

size = int(input("Enter house size: "))
bedrooms = int(input("Enter bedrooms: "))

prediction = model.predict([[size, bedrooms]])

print("Predicted Price:", prediction[0])