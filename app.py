# app.py
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_api():
    data = request.get_json()
    df = pd.DataFrame([data])
    prediction = model.predict(df)
    return jsonify({'prediction': int(prediction[0])})

@app.route('/predict_ui', methods=['POST'])
def predict_ui():
    form = request.form
    data = {
        'Pclass': int(form['Pclass']),
        'Sex': form['Sex'],
        'Age': float(form['Age']),
        'SibSp': int(form['SibSp']),
        'Parch': int(form['Parch']),
        'Fare': float(form['Fare']),
        'Embarked': form['Embarked'],
        'Title': form['Title']
    }
    df = pd.DataFrame([data])
    prediction = model.predict(df)
    result = "🎉 Survived!" if prediction[0] == 1 else "❌ Did not survive"
    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
 