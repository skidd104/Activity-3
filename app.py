# https://youtu.be/bluclMxiUkA
"""
Trained the machine learning model that predicts 
R&D Spend
Administration
Marketing Spend

"""

import numpy as np
from flask import Flask, request, render_template
import pickle

#Create an app object using the Flask class. 
app = Flask(__name__)

#Load the trained model. (Pickle file)
model = pickle.load(open('models/model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('michael.html')


@app.route('/predict',methods=['POST'])
def predict():

    int_features = [float(x) for x in request.form.values()] #Convert string inputs to float.
    features = [np.array(int_features)]  #Convert to the form [[a, b]] for input to the model
    prediction = model.predict(features)  # features Must be in the form [[a, b]]

    output = round(prediction[0], 2)

    return render_template('michael.html', prediction_text='Predicted Profit: {}'.format(output))

if __name__ == "__main__":
    app.run()