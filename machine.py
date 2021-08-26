## Intro into machine learning

"""
Supervised learning = prediction with a target variable
    - Predicting whether patients have heart disease
Unsupervised learning = finds the hidden patterns within the data with no target variable
    -
Supervised learning
    Classification = category variable (predict a colour, song genre, loan approval)
    Regression = continuous variable (predict price, income, how long someone looks for a job)

Unsupervised learning
    Clustering = identifying groups in your dataset
        K-means = specify number of clusters
        DBSCAN = specify what constitutes a cluster
    Association = finding relationships between observations
    Anomoly detection = detecting outliers that strongly differ from the others

Overfitting = evaluating the data
    - performs great on training data
    - performs poorly on testing data
    - Model memorizes training data and can't generalise learnings to new data (which is what we want)
"""
"""
## PREMIER LEAGUE MACHINE LEARNING PROJECT STEPS
    # 1. Clean the dataset
    # 2. Split into training and testing data (12 features and 1 target (winning team)
    # 3. Train 3 different classifiers on the data (logistic regression, support vector & XGBoost)
    # 4. Use the best classifier to predict who will win

df = pd.read_csv("results.csv", encoding='latin1')
df = df.dropna() # drops all the NaN values from the dataset
df = df.reset_index() #resets the index to 0
df = df.iloc[:,1:] # removing the first column (index) from df
#print (df.head(20))
#print(df.columns)
"""
#Index(['Season', 'DateTime', 'HomeTeam', 'AwayTeam',
#'FTHG' = Full Time Home Goals
#'FTAG' = Full Time Away Goals
#'FTR' = Full Time Result
#'HTHG', 'HTAG', 'HTR', 'Referee', 'HS', 'AS', 'HST', 'AST', 'HC', 'AC', 'HF', 'AF', 'HY', 'AY', 'HR', 'AR'
"""
# Q1. What is the win rate of the home team?
n_matches = df.shape[0]
n_features = df.shape[1] - 1
n_homewins = len(df[df.FTR == 'H'])
wins = (float(n_homewins) / n_matches * 100)
print('Total number of matches: {}'.format(n_matches))
print('Number of features: {}'.format(n_features))
print('Number of matches won by home team: {}'.format(n_homewins))
print('Win rate of home team: {:2f}%'.format(wins))
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler # used to standardise data
from sklearn.model_selection import train_test_split # to split data into train and test data
from sklearn import svm # support vector machine
from sklearn.metrics import accuracy_score #
#from IPython.display import display
from tabulate import tabulate

pd.options.display.width = None
pd.options.display.max_columns = None
pd.set_option('display.max_rows', 3000)
pd.set_option('display.max_columns', 3000)

data = pd.read_csv('diabetes.csv')
""" Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
    'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome' 
    768 rows, 9 columns 
    Outcome = 1 diabetic, 0 non-diabetic
"""
#print(data.head(10))

# getting the statistical data of df
#print(data.describe())

# get data on outcome for diabetes
"""
print(data['Outcome'].value_counts())
0    500
1    268
"""
# get the average data for the outcome patients
#print(data.groupby('Outcome').mean())

# SEPARATE DATA AND LABEL
X = data.drop(columns='Outcome', axis=1)
Y = data['Outcome']

#print(X)
#print(Y)

# PERFORM DATA STANDARDISATION. This step is important when features have a wide range of values.
    # It helps the machine learning model to make better predictions

scaler = StandardScaler()
scaler.fit(X)

standardized_data = scaler.transform(X)
#print(standardized_data)

X = standardized_data
Y = data['Outcome']

# TRAINING AND TESTING DATA SPLIT
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, stratify=Y, random_state=2)
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
input_data = (5,166,72,19,175,25.8,0.587,51)

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
    print('The patient is not diabetic')
else:
    print('The patient is diabetic')