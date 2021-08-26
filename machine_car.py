"""CAR PRICE PREDICTION SUPERVISED MACHINE LEARNING MODEL"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn import metrics
from sklearn.preprocessing import StandardScaler # used to standardise data
from sklearn.model_selection import train_test_split # to split data into train and test data
from sklearn import svm # support vector machine
from sklearn.metrics import accuracy_score #

pd.options.display.width = None
pd.options.display.max_columns = None
pd.set_option('display.max_rows', 3000)
pd.set_option('display.max_columns', 3000)

car_data = pd.read_csv('car data.csv')
"""
['Car_Name', 'Year', 'Selling_Price', 'Present_Price', 'Kms_Driven', 
'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner']
301 rows, 9 columns
"""
#print(car_data)
"""
print(car_data['Fuel_Type'].value_counts())
Petrol    239
Diesel     60
CNG         2
print(car_data['Transmission'].value_counts())
Manual       261
Automatic     40
print(car_data['Seller_Type'].value_counts())
Dealer        195
Individual    106
"""

# Changing the strings into numerical values
car_data.replace({"Fuel_Type": {'Petrol': 0, 'Diesel': 1, 'CNG': 2}}, inplace=True)
car_data.replace({"Seller_Type": {'Dealer': 0, 'Individual': 1}}, inplace=True)
car_data.replace({"Transmission": {'Automatic': 1, 'Manual': 0}}, inplace=True)
#print(car_data.head())

# Separate the dataset & label
X = car_data.drop(columns=['Selling_Price', 'Car_Name'], axis=1)
Y = car_data['Selling_Price']
#print(X)
#print(Y)

# TRAINING AND TESTING DATA SPLIT
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.1, random_state=2)
#print(X.shape, X_train.shape, X_test.shape)

# Loading the linear regression model
    # Linear regression works well with variables that are directly correlated
lin_reg = LinearRegression()
lin_reg.fit(X_train, Y_train)

## EVALUATE THE MODEL
# Use R squared for linear regression models on training data
X_train_prediction = lin_reg.predict(X_train)
error_score = metrics.r2_score(Y_train, X_train_prediction)
#print('R squared error score:', error_score)

X_test_prediction = lin_reg.predict(X_test)
error_score = metrics.r2_score(Y_test, X_test_prediction)
#print('R squared error score:', error_score)

"""
# Visualise the actual prices & predicted prices
plt.scatter(Y_train, X_train_prediction)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual Prices vs Predicted Prices')
#plt.show()

plt.scatter(Y_test, X_test_prediction)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual Prices vs Predicted Prices')
plt.show()
"""

## PERFORMING LASSO REGRESSION
    # Lasso Regression performs better with multiple features
lasso_reg = Lasso()
lasso_reg.fit(X_train, Y_train)

## EVALUATE THE MODEL
# Use R squared for linear regression models on training data
X_train_prediction = lasso_reg.predict(X_train)
error_score = metrics.r2_score(Y_train, X_train_prediction)
print('R squared train error:', error_score)

X_test_prediction = lin_reg.predict(X_test)
error_score = metrics.r2_score(Y_test, X_test_prediction)
print('R squared test error:', error_score)