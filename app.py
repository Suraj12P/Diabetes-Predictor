import pandas as pd
import numpy as np
from flask import Flask,request,render_template
import pickle
from sklearn.preprocessing import MinMaxScaler


app = Flask(__name__)

dataset = pd.read_csv('diabetes.csv')
dataset_X = dataset.iloc[:,[0, 1, 4, 6]].values

model = pickle.load(open('model.pkl','rb'))
sc = MinMaxScaler(feature_range = (0,1))
dataset_scaled = sc.fit_transform(dataset_X)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    
    inputfeatures  = [float(x) for x in request.form.values()]
    finalfeatures = [np.array(inputfeatures)]
    prediction = model.predict(sc.transform(finalfeatures))
    
    if prediction == 1:
        pred = "You have Diabetes,please consult a Doctor"
    else:
        pred = "You don't have Diabetes."
        
    output = pred
    
    
    return render_template('index.html',prediction_text=output)

    
    
    
    
    
    
    
    
    
    
    

if __name__ == "__main__":
    app.run(debug=True)