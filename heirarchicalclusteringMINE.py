import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd

dataset=pd.read_csv('Mall_Customers.csv')
x=dataset.iloc[:,[3,4]].values

import scipy.cluster.hierarchy as sch
den=sch.dendrogram(sch.linkage(x, method ='ward'))
plt.title('dendrogram')
plt.xlabel('customers')
plt.ylabel('euclidean_distances')
plt.show()

from sklearn.cluster import  AgglomerativeClustering 
hc=AgglomerativeClustering(n_clusters=5,affinity='euclidean', linkage='ward')
y_hc=hc.fit_predict(x)

plt.scatter(x[y_hc== 0,0], x[y_hc== 0,1], s=100, c= 'yellow', label= 'cluster1')
plt.scatter(x[y_hc== 1,0], x[y_hc== 1,1], s=100 ,c= 'red', label= 'cluster2')
plt.scatter(x[y_hc== 2,0], x[y_hc== 2,1], s=100 ,c= 'green' , label= 'cluster3')
plt.scatter(x[y_hc== 3,0], x[y_hc== 3,1], s=100 ,c= 'blue' , label= 'cluster4')
plt.scatter(x[y_hc== 4,0], x[y_hc== 4,1], s=100, c= 'black', label= 'cluster5')
plt.title('hierarchical_Clustering')
plt.xlabel('Annual_income') 
plt.ylabel('Spending_score')
plt.legend()
plt.show()
