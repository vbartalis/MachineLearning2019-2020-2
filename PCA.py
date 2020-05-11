# -*- coding: utf-8 -*-
"""
Created on Mon May  4 10:58:44 2020

@author: vbart
"""

import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from sklearn import decomposition as decomp
from numpy import linalg as LA;
import matplotlib.colors as col;  # importing coloring tools from MatPlotLib


dataset = pd.read_csv(r"G:\uni\gepiTanulas\hf\data\google_review\google_review_ratings.csv")
data = dataset.iloc[:,2:].values
target = dataset.iloc[:,1].values.round()
n = data.shape[0]
p = data.shape[1]
k = 6

color = ['blue','green','red','purple','orange','skyblue']
#targetName = ['0','1','2','3','4','5']
#%%
pca = decomp.PCA()
pca.fit(data)
fig = plt.figure(1)
plt.title('Explained variance ratio plot')
var_ratio = pca.explained_variance_ratio_
x_pos = np.arange(len(var_ratio))
plt.xticks(x_pos,x_pos+1)
plt.xlabel('Principal Components')
plt.ylabel('Variance')
plt.bar(x_pos,var_ratio, align='center', alpha=0.5)
plt.show()


#%%

pca = decomp.PCA(n_components=2);
pca.fit(data);
iris_pc = pca.transform(data);
class_mean = np.zeros((k,p));
for i in range(k):
    class_ind = [target==i][0].astype(int);
    class_mean[i,:] = np.average(data, axis=0, weights=class_ind);
    PC_class_mean = pca.transform(class_mean);    
    full_mean = np.reshape(pca.mean_,(1,p));
    PC_mean = pca.transform(full_mean);

figure = plt.figure(2);
plt.title('Dimension reduction of the Iris data by PCA');
plt.xlabel('PC1');
plt.ylabel('PC2');
plt.scatter(iris_pc[:,0],iris_pc[:,1],s=50, marker='x', c=target, cmap=col.ListedColormap(color),label='Datapoints');
plt.scatter(PC_class_mean[:,0],PC_class_mean[:,1],s=300,marker='P', c=np.arange(k),cmap=col.ListedColormap(color),label='Class means');
plt.scatter(PC_mean[:,0],PC_mean[:,1],s=50,c='black',label='Overall mean');
plt.legend();
plt.show();


#%%
comp = [0.70, 0.80, 0.90, 0.95]

for j in comp:
    pca = decomp.PCA(n_components=j);
    pca.fit(data);
    iris_pc = pca.transform(data);
    class_mean = np.zeros((k,p));
    for i in range(k):
        class_ind = [target==i][0].astype(int);
        class_mean[i,:] = np.average(data, axis=0, weights=class_ind);
        PC_class_mean = pca.transform(class_mean);    
        full_mean = np.reshape(pca.mean_,(1,p));
        PC_mean = pca.transform(full_mean);
    print("-----")
    print(j)
    #print(pca.explained_variance_ratio_)
    print(pca.n_components_)

# 70% - 9
# 80% - 12
# 90% - 16
# 95% - 19


















