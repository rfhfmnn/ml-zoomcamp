from flask import Flask #queremos que el ping pong sea un web service

app=Flask("ping") 
 
@app.route('/ping',methods=['GET']) #cuando se haga un get a la ruta ping, se ejecuta la funcion ping
def ping():
    return "PONG"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9696) # ejecuta la aplicación en el host y puerto especificados