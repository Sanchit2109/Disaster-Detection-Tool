from sklearn.ensemble import RandomForestClassifier
import joblib

X_train = [[0.2, 50], [0.8, 70], [0.4, 30], [0.9, 85]]  
y_train = [0, 1, 0, 1]  

model = RandomForestClassifier()
model.fit(X_train, y_train)
joblib.dump(model, "disaster_model.pkl")

from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("disaster_model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    features = np.array(data["features"]).reshape(1, -1)
    prediction = model.predict(features)[0]
    return jsonify({"risk_level": str(prediction)})

if __name__ == "__main__":
    app.run(debug=True, port=5000)