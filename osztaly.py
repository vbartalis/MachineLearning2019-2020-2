# -*- coding: utf-8 -*-
# Attribute 1 : Unique user id
# Attribute 2 : Average ratings on churches
# Attribute 3 : Average ratings on resorts
# Attribute 4 : Average ratings on beaches
# Attribute 5 : Average ratings on parks
# Attribute 6 : Average ratings on theatres
# Attribute 7 : Average ratings on museums
# Attribute 8 : Average ratings on malls
# Attribute 9 : Average ratings on zoo
# Attribute 10 : Average ratings on restaurants
# Attribute 11 : Average ratings on pubs/bars
# Attribute 12 : Average ratings on local services
# Attribute 13 : Average ratings on burger/pizza shops
# Attribute 14 : Average ratings on hotels/other lodgings
# Attribute 15 : Average ratings on juice bars
# Attribute 16 : Average ratings on art galleries
# Attribute 17 : Average ratings on dance clubs
# Attribute 18 : Average ratings on swimming pools
# Attribute 19 : Average ratings on gyms
# Attribute 20 : Average ratings on bakeries
# Attribute 21 : Average ratings on beauty & spas
# Attribute 22 : Average ratings on cafes
# Attribute 23 : Average ratings on view points
# Attribute 24 : Average ratings on monuments
# Attribute 25 : Average ratings on gardens

import pandas as pd;
import numpy as np; 
import matplotlib.pyplot as plt;
from sklearn import datasets as ds;
from sklearn import model_selection as ms;
from sklearn import tree, naive_bayes, metrics, neural_network;


ratings = pd.read_csv(r"G:\uni\gepiTanulas\hf\data\google_review\google_review_ratings.csv");
ratings = ratings.drop('User',axis=1);

# X = ratings.drop('Category 1',axis=1).round();
X = ratings.iloc[:, [7,21,22,23]].values;
Y = ratings['Category 1'].round();

X_train, X_test, y_train, y_test = ms.train_test_split(X, 
             Y, test_size=0.3, random_state=2019);


#%% 
#DecisionTreeClassifier

acc_train = np.zeros((30));                                                       
acc_test = np.zeros((30));  
crit = 'entropy';

for i in range(30):
    depth = i+1;
    classification_tree = tree.DecisionTreeClassifier(criterion=crit,max_depth=depth);

    classification_tree = classification_tree.fit(X_train, y_train);
    pred_tree_test = classification_tree.predict(X_test);
    pred_tree_train = classification_tree.predict(X_train);

    acc_test[i] = metrics.accuracy_score(y_test, pred_tree_test); 
    acc_train[i] = metrics.accuracy_score(y_train, pred_tree_train); 

plt.figure(1);
plt.title('Accuracy plot  with DecisionTreeClassifier');
plt.plot(acc_train, color='blue');
plt.plot(acc_test, color='red')
plt.show();


#%%

crit = 'gini';
depth = 14;
classifier = tree.DecisionTreeClassifier(criterion=crit,max_depth=depth);                                               
classifier.fit(X_train, y_train);

predy_train = classifier.predict(X_train);
predy_test = classifier.predict(X_test);
accuracy_train = metrics.accuracy_score(y_train, predy_train);
accuracy_test = metrics.accuracy_score(y_test, predy_test);
conf_mat_train = metrics.confusion_matrix(y_train, predy_train);
conf_mat_test = metrics.confusion_matrix(y_test, predy_test);

#%%
#neural_network.MLPClassifier

acc_train = np.zeros((30));                                                       
acc_test = np.zeros((30));  
activation = 'relu';

for i in range(30):
    size = i+1;
    neural = neural_network.MLPClassifier(hidden_layer_sizes=(size), activation=activation);
    neural.fit(X_train, y_train);
    
    pred_tree_train = neural.predict(X_train);
    pred_tree_test = neural.predict(X_test);

    acc_test[i] = metrics.accuracy_score(y_test, pred_tree_test); 
    acc_train[i] = metrics.accuracy_score(y_train, pred_tree_train); 

plt.figure(2);
plt.title('Accuracy plot with MLPClassifier');
plt.plot(acc_train, color='blue');
plt.plot(acc_test, color='red')
plt.show();


#%%

activation = 'relu';
size = 11;
neural = neural_network.MLPClassifier(hidden_layer_sizes=(size), activation=activation);                                             
neural.fit(X_train, y_train);

pred_tree_train = neural.predict(X_train);
pred_tree_test = neural.predict(X_test);

accuracy_train = metrics.accuracy_score(y_train, pred_tree_train);
accuracy_test = metrics.accuracy_score(y_test, pred_tree_test);
conf_mat_train = metrics.confusion_matrix(y_train, pred_tree_train);
conf_mat_test = metrics.confusion_matrix(y_test, pred_tree_test);

