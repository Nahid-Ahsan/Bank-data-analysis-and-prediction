from flask import Flask, render_template, request
import pandas as pd
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
import pickle
from model import JobEncoder, monthEncoder, scaler
import warnings
warnings.filterwarnings('ignore')



app = Flask(__name__)


# load dataset 
df = pd.read_excel("ML Assessment Dataset (Bank Data).xlsx")


# drop column "contact" which is useless
df = df.drop('contact', axis=1)
df = df.drop(df[df['poutcome'] == 'other'].index, axis = 0, inplace =False)
df = df.drop(df[df['education'] == 'unknown'].index, axis = 0, inplace =False)


monthEncoder = monthEncoder.fit(df['month'])
JobEncoder = JobEncoder.fit(df['job'])

# get unique values for categorical features
job_values = df['job'].unique()
marital_values = df['marital'].unique()
education_values = df['education'].unique()
default_values = df['default'].unique()
housing_values = df['housing'].unique()
loan_values = df['loan'].unique()
month_values = df['month'].unique()
poutcome_values = df['poutcome'].unique()
model_values = ['Decision Tree', 'Naive Bayes']

@app.route('/')
def home():
    return render_template('index.html', job_values=job_values, marital_values=marital_values,
                           education_values=education_values, default_values=default_values,
                           housing_values=housing_values, loan_values=loan_values,
                           month_values=month_values,
                           poutcome_values=poutcome_values,
                           model_values = model_values,
                           )


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        age = float(request.form['age'])
        job = request.form['job']
        marital = request.form['marital']
        education = request.form['education']
        default = request.form['default']
        balance = float(request.form['balance'])
        housing = request.form['housing']
        loan = request.form['loan']
        day = int(request.form['day'])
        month = request.form['month']
        duration = float(request.form['duration'])
        campaign = int(request.form['campaign'])
        pdays = int(request.form['pdays'])
        previous = int(request.form['previous'])
        poutcome = request.form['poutcome']
        model = request.form['model']



        # load pre-trained machine learning model
        if model == 'Decision Tree':
            model = pickle.load(open('model_tree.pkl', 'rb'))
        else:
            model = pickle.load(open('model_nb.pkl', 'rb'))

        # create a DataFrame with the user input
        input_data = pd.DataFrame({
            'age': [age],
            'job': [job],
            'marital': [marital],
            'education': [education],
            'default': [default],
            'balance': [balance],
            'housing': [housing],
            'loan': [loan],
            'day': [day],
            'month': [month],
            'duration': [duration],
            'campaign': [campaign],
            'pdays': [pdays],
            'previous': [previous],
            'poutcome': [poutcome],
           
        })

        # preprocess input data
        input_data['duration'] = round(input_data['duration'] / 60, 2)
        input_data['month'] = monthEncoder.transform([input_data['month']])[0]
        input_data['job'] = JobEncoder.transform([input_data['job']])[0]
        input_data['default'] = input_data['default'].apply(lambda x: 0 if x == 'no' else 1)
        input_data['housing'] = input_data['housing'].apply(lambda x: 0 if x == 'no' else 1)
        input_data['loan'] = input_data['loan'].apply(lambda x: 0 if x == 'no' else 1)
        input_data['marital'] = input_data['marital'].map({'married': 1, 'single': 2, 'divorced': 3})
        input_data['education'] = input_data['education'].map({'primary': 1, 'secondary': 2, 'tertiary': 3})
        input_data['poutcome'] = input_data['poutcome'].map({'unknown': 1, 'failure': 2, 'success': 3})


        # scale the numerical features
        feature_scale = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
        inputScale = scaler.transform(input_data[feature_scale])
        input_data[feature_scale] = inputScale


        # predictions 
        prediction = model.predict(input_data)
        print(prediction)
        if prediction[0] == 0:
            res = "NO"
        else:
            res = "YES"

        return render_template('result.html', prediction=res)

if __name__ == '__main__':
    app.run(debug=True)
