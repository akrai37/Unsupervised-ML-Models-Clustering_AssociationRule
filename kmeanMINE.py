import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('Mall_Customers.csv')
x=dataset.iloc[:, [3,4]].values

from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
from matplotlib.pyplot import xticks    
plt.plot(range(1,11), wcss)
xticks(range(10), range(10))
plt.title('Elbow_Method')
plt.xlabel('no. of clusters') 
plt.ylabel('wcss')
plt.show()   

kmeans=KMeans(n_clusters= 4, init='k-means++', max_iter=300, n_init=10, random_state=0)
y_kmeans= kmeans.fit_predict(x)

plt.scatter(x[y_kmeans==0,0], x[y_kmeans==0,1], s=100 , c='cyan', label='cluster1')
plt.scatter(x[y_kmeans==1,0], x[y_kmeans==1,1], s=100 , c='green', label='cluster2')
plt.scatter(x[y_kmeans==2,0], x[y_kmeans==2,1], s=100 , c='red', label='cluster3')
plt.scatter(x[y_kmeans==3,0], x[y_kmeans==3,1], s=100 , c='blue', label='cluster4')
plt.scatter(x[y_kmeans==4,0], x[y_kmeans==4,1], s=100 , c='magenta', label='cluster5')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=300 , c='yellow', label='centroids')
plt.title('clusters of clients')
plt.xlabel('Annual_income') 
plt.ylabel('Spending_score')
plt.legend()
plt.show()
















