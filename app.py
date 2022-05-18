import pickle
from flask import Flask
from flask import request
from flask import jsonify
from train import Perceptron
import joblib

app = Flask(__name__)

@app.route('/api/v1.0/predict', methods=['GET'])
def get_prediction():

    sepal_length = float(request.args.get('sl'))

    petal_length = float(request.args.get('pl'))

    features = [sepal_length, petal_length]

    model = joblib.load('model.pkl')

    predicted_class = int(model.predict([features]))

    return jsonify(features=features, predicted_class=predicted_class)

if __name__ == '__main__':
    app.run(port=3333,host='0.0.0.0')
