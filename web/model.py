import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
import pickle
import warnings
warnings.filterwarnings('ignore')


df = pd.read_excel("ML Assessment Dataset (Bank Data).xlsx")


df['job'] = df[['job']].replace(['unknown'],'other')
df = df.drop('contact', axis=1)
df['duration'] = df['duration'].apply(lambda n:n/60).round(2)

monthEncoder = preprocessing.LabelEncoder()
df['month'] = monthEncoder.fit_transform(df['month'])

JobEncoder = preprocessing.LabelEncoder()
df['job'] = monthEncoder.fit_transform(df['job'])

df = df.drop(df[df['poutcome'] == 'other'].index, axis = 0, inplace =False)
df = df.drop(df[df['education'] == 'unknown'].index, axis = 0, inplace =False)
df['y'] = df['y'].apply(lambda x: 0 if x == 'no' else 1)
df['default'] = df['default'].apply(lambda x: 0 if x == 'no' else 1)
df['housing'] = df['housing'].apply(lambda x: 0 if x == 'no' else 1)
df['loan'] = df['loan'].apply(lambda x: 0 if x == 'no' else 1)
df['marital'] = df['marital'].map({'married': 1, 'single': 2, 'divorced': 3})
df['education'] = df['education'].map({'primary': 1, 'secondary': 2, 'tertiary': 3})
df['poutcome'] = df['poutcome'].map({'unknown': 1, 'failure': 2, 'success': 3})

feature_scale = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
scaler=StandardScaler()
scale = scaler.fit_transform(df[feature_scale])
df[feature_scale] = scale

X = df.drop('y', axis=1)
Y = df['y']

X_train, X_test, Y_train, Y_test= train_test_split(X, Y, test_size=0.20, random_state=7)


# DecisionTree
clf_tree = DecisionTreeClassifier(criterion='entropy',
                                  max_depth=2)
clf_tree.fit(X_train, Y_train)

# Prediction 
# TreePredictions = clf_tree.predict(X_test)
# # Accuracy Score 
# print(accuracy_score(Y_test, TreePredictions))

pickle.dump(clf_tree, open('model_tree.pkl', 'wb'))



clf_nb = GaussianNB()
clf_nb.fit(X_train, Y_train)

# NbPredictions = clf_nb.predict(X_test)
# # Accuracy Score 
# print(accuracy_score(Y_test, NbPredictions))

pickle.dump(clf_nb, open('model_nb.pkl', 'wb'))

