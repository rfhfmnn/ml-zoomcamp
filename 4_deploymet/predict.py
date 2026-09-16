import pickle
from flask import Flask
from flask import request
from flask import jsonify

input_file = "model_C=1.0.bin"

with open(input_file, "rb") as f_in:  # abrir el archivo de entrada en modo binario
    dv, model = pickle.load(f_in)

customer = {
    "gender": "female",
    "seniorcitizen": 0,
    "partner": "yes",
    "dependents": "no",
    "phoneservice": "no",
    "multiplelines": "no_phone_service",
    "internetservice": "dsl",
    "onlinesecurity": "no",
    "onlinebackup": "yes",
    "deviceprotection": "no",
    "techsupport": "no",
    "streamingtv": "no",
    "streamingmovies": "no",
    "contract": "month-to-month",
    "paperlessbilling": "yes",
    "paymentmethod": "electronic_check",
    "tenure": 1,
    "monthlycharges": 29.85,
    "totalcharges": 29.85
}

app=Flask("churn")
@app.route("/predict",methods=["POST"]) # define the route for the ping endpoint. POST method is used to send data to the server.

def predict():
    customer = request.get_json()

    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0, 1]
    churn=y_pred>=0.5

    result={
        "churn_probability":float(y_pred),
        "churn": bool(churn) #se hace boolean para que sea True o False en lugar de 1 o 0 y para que sea compatible con JSON, ya que JSON no tiene un tipo de datos booleano nativo.
    }

    return jsonify(result)

if __name__=="__main__":
    app.run(debug=True,host="0.0.0.0",port=9696)