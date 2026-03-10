from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

# load trained model
model = joblib.load("model.pkl")

# home page
@app.route("/")
def home():
    return render_template("index.html")

# prediction route
@app.route("/predict", methods=["POST"])
def predict():

    size = float(request.form["size"])
    bedrooms = float(request.form["bedrooms"])
    bathrooms = int(request.form['bathrooms'])
floors = int(request.form['floors'])
age = int(request.form['age'])
parking = int(request.form['parking'])

prediction = model.predict([[area, bedrooms, bathrooms, floors, age, parking]])

result = f"Predicted House Price: ${prediction[0]:,.2f}"
    prediction = model.predict([[size, bedrooms]])

    result = f"Here is your total predicted house price: ₹ {round(prediction[0],2)}"

    return render_template("index.html", prediction_text=result)


if __name__ == "__main__":
    app.run(debug=True)