"""
Created on Sun Jun 21 19:41:17 2020

@author: aman kumar
"""

"""In this problem you are given a Diabetes Data set consisting of following features -
['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
and your task is to predict whether a person is suffering from diabetes or not (Binary
Classification)
Tasks
1) Perform necessary data visualization and preprocessing.
2) Classification Task, classify a person as 0 or 1 (Diabetic or Not) using K-Nearest Neighbors
classifier/Logistic regression"""


#importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#importing the datasets
X_train = pd.read_csv('Diabetes_XTrain.csv')
X_test = pd.read_csv('Diabetes_Xtest.csv')
y_train = pd.read_csv('Diabetes_YTrain.csv')

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#Fitting K-NN classifier to the training set  
from sklearn.neighbors import KNeighborsClassifier  
classifier= KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2 )  
classifier.fit(X_train, y_train)

#predicting the test results
y_pred = classifier.predict(X_test)

