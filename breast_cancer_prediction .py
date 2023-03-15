# -*- coding: utf-8 -*-

import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

from sklearn import datasets
data = datasets.load_breast_cancer()

data.feature_names

data.target_names

df = pd.DataFrame(data.data, columns=data.feature_names)

df["Target"] = data.target

df.head()

df["Target"].value_counts()

"""Thus we have ,
    1-->Benign
    0-->Malignant
"""

df.isnull().sum()

df.groupby(df["Target"]).mean()

df.describe()

df.shape

#plotting the histogram for each feature
for i in range(len(data.feature_names)):
  plt.figure(figsize=(4,4))
  plt.title('Histogram')
  plt.xlabel(data.feature_names[i])
  plt.ylabel('Frequency')
  plt.hist(df[data.feature_names[i]], bins=20)
  plt.show()

#let's plot a scatter vector towards target variable
for i in range(len(data.feature_names)):
  plt.figure(figsize=(4,4))
  plt.title('Scatter')
  plt.xlabel(data.feature_names[i])
  plt.ylabel('Target')
  plt.scatter(df[data.feature_names[i]], df["Target"])
  plt.show()

#check for the corelation using heatmap
plt.figure(figsize=(10,10))
plt.title('Correlation Matrix')
plt.xticks(range(len(data.feature_names)), data.feature_names, rotation=90)
plt.yticks(range(len(data.feature_names)), data.feature_names)
plt.imshow(df.corr(), interpolation='none', cmap='Blues')
plt.colorbar()
plt.show()

#Lets split the data into train and test
X = df.drop("Target", axis=1)
y = df["Target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=10)

#lets evaluate the prediction using Ensemble SVM classifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# train a SVM classifier
classifier = SVC(kernel='rbf', C=1, gamma='scale', random_state=12)
classifier.fit(X_train, y_train)



# make predictions on the test set
y_pred = classifier.predict(X_test)

# calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc_roc = roc_auc_score(y_test, y_pred)

# print the evaluation metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)
print("AUC ROC:", auc_roc)

from sklearn.model_selection import GridSearchCV

# define the hyperparameter grid to search over
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': ['scale', 'auto'], 'kernel': ['rbf', 'linear']}

# perform grid search to find the best hyperparameters
grid_search = GridSearchCV(estimator=classifier, param_grid=param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

# print the best hyperparameters and their corresponding score
print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)

"""AFTER GRIDSEARCH-CV WE DEFINE THE BEST FIT PARAMETERS"""

# train a SVM classifier
classifier = SVC(kernel='linear', C=100, gamma='scale', random_state=1)
classifier.fit(X_train, y_train)



# make predictions on the test set
y_pred = classifier.predict(X_test)

# calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc_roc = roc_auc_score(y_test, y_pred)

# print the evaluation metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)
print("AUC ROC:", auc_roc)

