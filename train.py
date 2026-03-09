import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib
file_path = input("Enter dataset path: ")
data = pd.read_csv(file_path)

print("Dataset Preview:")
print(data.head())

# automatically select features and target
X = data.iloc[:, :-1]   # all columns except last
y = data.iloc[:, -1]    # last column

# split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# create model
model = LinearRegression()

# train model
model.fit(X_train, y_train)

# check accuracy
score = model.score(X_test, y_test)
print("Model Accuracy:", score)

# save model
joblib.dump(model, "model.pkl")
print("Model trained and saved!")