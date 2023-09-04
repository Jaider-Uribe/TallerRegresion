from flask import Flask, render_template, request
import joblib
import os


model_path = os.path.join(os.path.dirname( __file__), 'models', 'modelo.pkl')
model = joblib.load(model_path) 

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')  


@app.route('/predict', methods=['POST'])
def predict():
    sepalo_Largo_cm = float(request.form['largoSepalo'])
    sepalo_ancho_cm = float(request.form['anchoSepalo'])
    petalo_largo_cm = float(request.form['largoPetalo'])
    petalo_ancho_cm = float(request.form['anchoPetalo'])
    
  
    pred_probabilities = model.predict_proba([[sepalo_Largo_cm, sepalo_ancho_cm, petalo_largo_cm, petalo_ancho_cm]])
    
    Iris = model.classes_

    mensaje = ""
    for i, flor in enumerate(Iris):
        prob = pred_probabilities[0, i] * 100
        mensaje += f"Probabilidad de {flor}: {round(prob, 2)}% <br/>"

    return render_template('result.html', pred=mensaje)

if __name__ == '__main__':
    app.run()