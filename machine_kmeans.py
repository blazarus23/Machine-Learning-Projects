"""CUSTOMER SEGMENTATION K-MEANS CLUSTERING MACHINE LEARNING MODEL"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
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

cust_data = pd.read_csv('Mall_Customers.csv')
"""
['CustomerID', 'Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']
200 rows, 5 columns
"""
#print(cust_data.info())
#print(cust_data.describe())

# Selecting the values from income and spending columns
X = cust_data.iloc[:,3:5].values
#print(X)

# Finding the optimal number of cluster using WCSS (Within Clusters Sum of Squares)

wcss = []

for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)

    wcss.append(kmeans.inertia_)

# Plot an elbow graph to visualise number of clusters and WCSS values
sns.set()
plt.plot(range(1,11), wcss)
plt.title('The Elbow Point Graph')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS Value')
#plt.show()

# The optimum number of clusters = 5
# Training the KMeans clustering unsupervised learning model

kmeans = KMeans(n_clusters=5, init='k-means++', random_state=0)

# Return a label for each data point based on their cluster
Y = kmeans.fit_predict(X)
#print(Y)

# Visualise all the clusters & their centroids
plt.figure(figsize=(8,8))
plt.scatter(X[Y==0,0], X[Y==0,1], s=50, c='green', label='Cluster 1')
    # All the first zeros (4 of them) represent annual income (first column) in X
    # ALl the second 1's (4 of them) represent spending score (second column) in X
    # The first value in each represents what number cluster (0 to 4)
    # s = size of the dot
    # c = colour of the dot
plt.scatter(X[Y==1,0], X[Y==1,1], s=50, c='red', label='Cluster 2')
plt.scatter(X[Y==2,0], X[Y==2,1], s=50, c='yellow', label='Cluster 3')
plt.scatter(X[Y==3,0], X[Y==3,1], s=50, c='blue', label='Cluster 4')
plt.scatter(X[Y==4,0], X[Y==4,1], s=50, c='purple', label='Cluster 5')

# Plot the centroids
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=100, c='black', label='Centroid')
    # [:,0] = represents the X axis of the centroids
    # [:,1] = represents the Y axis of the centroids
plt.title('Customer Groups')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score (1-100)')
plt.show()
