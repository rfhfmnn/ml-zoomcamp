import pickle

from flask import Flask
from flask import request
from flask import jsonify


model_file = 'model_C=1.0.bin'

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

app = Flask('churn')

@app.route('/predict', methods=['POST']) #seria POST porque le estamos mandando datos al modelo para que nos haga la prediccion
def predict():
    customer = request.get_json() # el request es un objeto de flask que tiene toda la informacion de la request que se hizo al endpoint, en este caso es un POST, y el metodo get_json() nos devuelve el json que se mando en la request

    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0, 1]
    churn = y_pred >= 0.5
    result = {
        'churn_probability': float(y_pred),
        'churn': bool(churn)
    }

    return jsonify(result) #jsonify es un metodo de flask que convierte un diccionario en un json y lo devuelve como respuesta a la request


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)