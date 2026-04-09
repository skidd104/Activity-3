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
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.impute import SimpleImputer

#Create an app object using the Flask class. 
app = Flask(__name__)

#Load the trained model. (Pickle file)
model = pickle.load(open('models/model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('michael.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        rd_spend = float(request.form.get('R&D_Spend'))
        admin_spend = float(request.form.get('Administration'))
        market_spend = float(request.form.get('Marketing_Spend'))
        
  
        features = np.array([[rd_spend, admin_spend, market_spend]])
        
        prediction = model.predict(features)
        output = "{:,.2f}".format(prediction[0]) 

        return render_template('michael.html', prediction_text=f'Estimated Profit: ${output}')
    
    except Exception as e:
        return render_template('michael.html', prediction_text="Error in calculation. Please check inputs.")

@app.route('/insights')
def insights():

    df = pd.read_csv('./files_training/Data_Startups.csv') 

    plt.figure(figsize=(12, 10))
    sns.heatmap(df.drop('City', axis=1).corr(), annot=True, cmap='RdYlGn', center=0, annot_kws={"size": 12})
    plt.title('Feature Correlation: What drives Profit?', pad=20, size=15)
    
    if not os.path.exists('static/images'):
        os.makedirs('static/images')
        
    plt.savefig('static/images/heatmap.png')
    plt.close()
    
    return render_template('insights.html')

@app.route('/retrain', methods=['POST'])
def retrain():
    try:
        file = request.files['database_file']
        algo_choice = request.form.get('algorithm')
        
        if not file: return "No file", 400
        
        new_df = pd.read_csv(file)
        X = new_df[['R&D Spend', 'Administration', 'Marketing Spend']].values
        y = new_df['Profit'].values
        
        # Create the imputer to fill NaNs with the mean of the column
        imputer = SimpleImputer(strategy='mean')
        X = imputer.fit_transform(X) # This fills the holes in X

        if algo_choice == 'random_forest':
            new_model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif algo_choice == 'decision_tree':
            new_model = DecisionTreeRegressor()
        else:
            new_model = LinearRegression()

        new_model.fit(X, y)
        
        with open('models/model.pkl', 'wb') as f:
            pickle.dump(new_model, f)
            
        global model
        model = new_model
        
        return render_template('michael.html', prediction_text=f"Model updated using {algo_choice.replace('_', ' ').title()}!")
    
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)