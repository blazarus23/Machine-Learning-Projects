"""LOAN PREDICTION SUPERVISED MACHINE LEARNING MODEL"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler # used to standardise data
from sklearn.model_selection import train_test_split # to split data into train and test data
from sklearn import svm # support vector machine
from sklearn.metrics import accuracy_score #

pd.options.display.width = None
pd.options.display.max_columns = None
pd.set_option('display.max_rows', 3000)
pd.set_option('display.max_columns', 3000)

loan_data = pd.read_csv('train_u6lujuX_CVtuZ9i (1).csv')

# Column names & DF info
"""
'Loan_ID', 'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'ApplicantIncome', 
'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Property_Area', 'Loan_Status'
614 rows, 13 columns
"""
# printing the statistical information on dataset
#print(loan_data.describe())

"""
print(loan_data['Loan_Status'].value_counts())
Y    422
N    192
"""

# count the total number of missing/NaN in dataset
#print(loan_data.isnull().sum())

# drop all the missing NaN values from dataset
loan_data.dropna(axis='index', how='any', inplace=True)
#print(loan_data.isnull().sum())

# label encoding, changing Y/N to 0/1 (loan status and 3+ to 4 (dependents)
loan_data.replace({"Loan_Status": {'Y': 1, 'N': 0}}, inplace=True)
loan_data.replace({"Dependents": {'3+': 4}}, inplace=True)
#print(loan_data['Dependents'].value_counts())

## DATA VISUALISATION
# Check the relationship of education & loan status
sns.countplot(x='Education', hue='Loan_Status', data=loan_data)
#plt.show()

# Visualise martial & loan status
sns.countplot(x='Married', hue='Loan_Status', data=loan_data)
#plt.show()

# Convert all string columns to integers values: Yes = 1, No = 0
# Need to perform this step because the model will not understand the string columns properly
loan_data.replace({"Gender": {'Male': 1, 'Female': 0}}, inplace=True)
loan_data.replace({"Married": {'Yes': 1, 'No': 0}}, inplace=True)
loan_data.replace({"Education": {'Graduate': 1, 'Not Graduate': 0}}, inplace=True)
loan_data.replace({"Self_Employed": {'Yes': 1, 'No': 0}}, inplace=True)
loan_data.replace({"Gender": {'Male': 1, 'Female': 0}}, inplace=True)
loan_data.replace({"Property_Area": {'Rural': 0, 'Semiurban': 1, 'Urban': 2}}, inplace=True)
#print(loan_data)

# Separate the dataset & label
X = loan_data.drop(columns=['Loan_Status', 'Loan_ID'], axis=1)
Y = loan_data['Loan_Status']
#print(X)
#print(Y)

# TRAINING AND TESTING DATA SPLIT
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.1, stratify=Y, random_state=2)
#print(X.shape, X_train.shape, X_test.shape)

## TRAINING THE MODEL
classifier = svm.SVC(kernel='linear')

# Training the support vector machine classifier
classifier.fit(X_train, Y_train)

## EVALUATE THE MODEL
# Find the accuracy score on the training data
X_train_prediction = classifier.predict(X_train)
trainingdata_accuracy = accuracy_score(X_train_prediction, Y_train)

#print('Accuracy score of the training data : ', trainingdata_accuracy)

# Find the accuracy score on the test data
X_test_prediction = classifier.predict(X_test)
testdata_accuracy = accuracy_score(X_test_prediction, Y_test)

#print('Accuracy score of the test data : ', testdata_accuracy)

## MAKING A PREDICTIVE SYSTEM BASED ON FEATURES
scaler = StandardScaler()
scaler.fit(X)

input_data = (1,1,1,1,0,4583,1508,128,360,1,0)

# change data to a numpy array
input_data_numpy = np.asarray(input_data)

#reshape the array data as we are predicting for one instance
input_data_reshaped = input_data_numpy.reshape(1,-1)

# standardize the input data
std_data = scaler.transform(input_data_reshaped)
print(std_data)

prediction = classifier.predict(std_data)
print(prediction)

if (prediction[0] == 0):
    print('The loan is rejected')
else:
    print('The loan is approved')

#print(loan_data)