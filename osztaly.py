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
import itertools;


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Greens):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

#%%
ratings_url = 'https://raw.githubusercontent.com/vbartalis/MachineLearning2019-2020-2/master/data/google_review/google_review_ratings.csv'
ratings = pd.read_csv(ratings_url);
ratings = ratings.drop('User',axis=1);

# attributes_url = 'https://raw.githubusercontent.com/vbartalis/MachineLearning2019-2020-2/master/data/google_review/attribute_names.csv'
# attributes = pd.read_csv(attributes_url, dtype=str);
# attr = attributes.drop('Attribute 1',axis=1);

target = [0,1,2,3,4,5]


# X = ratings.iloc[:, [7,21,22,23]].values.round()
X = ratings.drop('Category 2',axis=1).round();
Y = ratings['Category 2'].round()

X_train, X_test, y_train, y_test = ms.train_test_split(X, 
             Y, test_size=0.3, random_state=2019);


# attr = attr.iloc[0,[7,21,22,23]]
# attr = attr.transpose().values.tolist()
# attribute_names = [];
# for i in attr:
#     name = i.replace("Average ratings on ", "")
#     attribute_names.append(name)

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
depth = 13;
classifier = tree.DecisionTreeClassifier(criterion=crit,max_depth=depth);                                               
classifier.fit(X_train, y_train);

predy_train = classifier.predict(X_train);
predy_test = classifier.predict(X_test);
accuracy_train = metrics.accuracy_score(y_train, predy_train);
accuracy_test = metrics.accuracy_score(y_test, predy_test);
conf_mat_train = metrics.confusion_matrix(y_train, predy_train);
conf_mat_test = metrics.confusion_matrix(y_test, predy_test);

plt.figure(2);
plot_confusion_matrix(conf_mat_train, classes=target,
    title='Confusion matrix for training dataset (DecisionTreeClassifier)');
plt.show();

plt.figure(3);
plot_confusion_matrix(conf_mat_test, classes=target,
    title='Confusion matrix for testing dataset (DecisionTreeClassifier)');
plt.show();
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
    

plt.figure(4);
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



plt.figure(5);
plot_confusion_matrix(conf_mat_train, classes=target,
    title='Confusion matrix for training dataset (DecisionTreeClassifier)');
plt.show();

plt.figure(6);
plot_confusion_matrix(conf_mat_test, classes=target,
    title='Confusion matrix for testing dataset (DecisionTreeClassifier)');
plt.show();
