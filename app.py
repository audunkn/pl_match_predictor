import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
np.set_printoptions(precision=1)

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    prediction.astype(str)
    #output = prediction
    if int(prediction)== 1: 
            pred ='DRAW/AWAY '
    else: 
            pred ='HOME'
    
    return render_template('index.html', prediction_text='Most likely match result: {}'.format(pred))
    

#@app.route('/results',methods=['POST'])
#def results():

 #   data = request.get_json(force=True)
  #  prediction = model.predict([np.array(list(data.values()))])
   
    
   # output = prediction[0]
    #return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)