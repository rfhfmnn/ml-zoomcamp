from flask import Flask

app=Flask("Ping")

@app.route('/ping',methods=['GET']) # define the route for the ping endpoint. GET method is used to retrieve data from the server.
def ping():
    return "PONG"

if __name__=='__main__':
    app.run(debug=True,host='0.0.0.0',port=9696)