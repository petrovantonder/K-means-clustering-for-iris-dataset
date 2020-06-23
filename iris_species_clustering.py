import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.cluster import KMeans
from sklearn import preprocessing

data = pd.read_csv('iris-dataset.csv')

print(data)

plt.scatter(data['sepal_length'],data['sepal_width'])
plt.xlabel('Lenght of sepal')
plt.ylabel('Width of sepal')
plt.show()

# Clustering (unscaled)
x = data.copy()
kmeans = KMeans(2)
kmeans.fit(x)

clusters = data.copy()
clusters['cluster_pred'] = kmeans.fit_predict(x)

plt.scatter(clusters['sepal_length'], clusters['sepal_width'], c=clusters['cluster_pred'], cmap='rainbow')
plt.show()

#Standardizing the data
x_scaled = preprocessing.scale(data)
print(x_scaled)

#Clustering with scaled data
kmeans_scaled = KMeans(2)
kmeans_scaled.fit(x_scaled)

clusters_scaled = data.copy()
clusters_scaled['cluster_pred'] = kmeans_scaled.fit_predict(x_scaled)

plt.scatter(clusters_scaled['sepal_length'], clusters_scaled['sepal_width'], c=clusters_scaled['cluster_pred'], cmap='rainbow')
plt.show()

wcss = []
cl_num = 10
for i in range (1,cl_num):
    kmeans= KMeans(i)
    kmeans.fit(x_scaled)
    wcss_iter = kmeans.inertia_
    wcss.append(wcss_iter)
print(wcss)

number_clusters = range(1,cl_num)
plt.plot(number_clusters, wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('Within-cluster Sum of Squares')
plt.show()

#Testing the elbow method to determine the optimal clusters for data set
kmeans_2 = KMeans(2)
kmeans_2.fit(x_scaled)

clusters_2 = x.copy()
clusters_2['cluster_pred'] = kmeans_2.fit_predict(x_scaled)

plt.scatter(clusters_2['sepal_length'], clusters_2['sepal_width'], c=clusters_2['cluster_pred'], cmap='rainbow')
plt.show()

kmeans_3 = KMeans(3)
kmeans_3.fit(x_scaled)

clusters_3 = x.copy()
clusters_3['cluster_pred'] = kmeans_3.fit_predict(x_scaled)
plt.scatter(clusters_3['sepal_length'], clusters_3['sepal_width'], c=clusters_3['cluster_pred'], cmap='rainbow')
plt.show()

kmeans_5 = KMeans(5)
kmeans_5.fit(x_scaled)

clusters_5 = x.copy()
clusters_5['cluster_pred'] = kmeans_5.fit_predict(x_scaled)

plt.scatter(clusters_5['sepal_length'], clusters_5['sepal_width'], c=clusters_5['cluster_pred'], cmap='rainbow')
plt.show()

#Comparing solution to the original data set
real_data = pd.read_csv('iris-with-answers.csv')
real_data['species'].unique()

# Using the map function to change any 'yes' values to 1 and 'no'values to 0. 
real_data['species'] = real_data['species'].map({'setosa': 0, 'versicolor': 1, 'virginica': 2})
print(real_data.head())

plt.scatter(real_data['sepal_length'], real_data['sepal_width'], c=real_data['species'], cmap='rainbow')
plt.show()

plt.scatter(real_data['petal_length'], real_data['petal_width'], c=real_data['species'], cmap='rainbow')
plt.show()
plt.scatter(clusters_3['sepal_length'], clusters_3['sepal_width'], c=clusters_3['cluster_pred'], cmap='rainbow')
plt.show()
plt.scatter(clusters_3['petal_length'], clusters_3['petal_width'], c= clusters_3 ['cluster_pred'], cmap = 'rainbow')
plt.show()