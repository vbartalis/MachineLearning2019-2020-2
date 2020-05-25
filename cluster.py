# -*- coding: utf-8 -*-
"""
Created on Sat May  2 21:14:59 2020

@author: vbart
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn import metrics;

url = 'https://raw.githubusercontent.com/vbartalis/MachineLearning2019-2020-2/master/data/google_review/google_review_ratings.csv'
dataset = pd.read_csv(url)
X = dataset.iloc[:, [1,10]].values

#%%
# #Elbow method
# maxClassters = 30
# wcss =[]
# for i in range (1,maxClassters):
#     kmeans = KMeans(n_clusters = i,init = 'k-means++',max_iter = 300,n_init = 10, random_state = 0)
#     kmeans.fit(X)
#     wcss.append(kmeans.inertia_)
# fig = plt.figure(1);
# plt.plot(range(1,maxClassters),wcss)
# plt.title('Elbow method for the musics')
# plt.xlabel('Number of clusters')
# plt.ylabel('wcss')
# plt.show()

#%%
Max_K = 30;
SSE = np.zeros((Max_K-2));
DB = np.zeros((Max_K-2));
for i in range(Max_K-2):
    n_c = i+2;
    kmeans = KMeans(n_clusters=n_c, random_state=2020);
    kmeans.fit(X);
    ypred = kmeans.labels_;
    SSE[i] = kmeans.inertia_;
    DB[i] = metrics.davies_bouldin_score(X,ypred);
    
fig = plt.figure(2);
plt.title('Sum of squares of error curve');
plt.xlabel('Number of clusters');
plt.ylabel('SSE');
plt.plot(np.arange(2,Max_K),SSE, color='red')
plt.show();

fig = plt.figure(3);
plt.title('Davies-Bouldin score curve');
plt.xlabel('Number of clusters');
plt.ylabel('DB index');
plt.plot(np.arange(2,Max_K),DB, color='blue')
plt.show();

#%%
clasters = 6
kmeans = KMeans(n_clusters = clasters,init = 'k-means++',max_iter = 300, n_init = 10, random_state = 0)
y = kmeans.fit_predict(X)
y = np.reshape(y, (-1,1))
results = np.concatenate((X,y),axis = 1)
header =['Church','Bar','Cluster']
temp = pd.DataFrame(data = results, columns = header)
color = ['blue','green','red','purple','orange','skyblue']
fig = plt.figure(4);
for i in range (0,clasters):
    
    show = temp[(temp['Cluster']==i)]
    # plt.scatter(show['Church'], show['Bar'] ,marker='x', color = color[i])
    plt.scatter(show['Church'], show['Bar'] ,marker='x')
    
plt.xlabel('Church review')
plt.ylabel('Bar review')
plt.grid()
plt.axis('scaled')
# plt.xlim(-0.5,5.5)
# plt.ylim(-0.5,5.5)
plt.show()
