#!/usr/bin/env python3

import pandas as pd

seed = 666

data = pd.read_csv('heart_failure.csv')
print(data.head())

x = data.iloc[:, 0:-1].values # features: from first to next-to-last column
y = data.iloc[:, -1].values # label: last column

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 1/3, random_state = seed)

x_train_2, x_val, y_train_2, y_val = train_test_split(
    x_train, y_train, test_size = 1/3, random_state = seed)

# feature scaling

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
x_train_2 = scaler.fit_transform(x_train_2)
x_val = scaler.transform(x_val)

dic = {} # result dictionary

from sklearn.metrics import accuracy_score

# logistic regression

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(random_state = seed)
lr.fit(x_train_2, y_train_2)

dic["logistic_regression"] = accuracy_score(y_val, lr.predict(x_val))

# k-nn

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 7)
knn.fit(x_train_2, y_train_2)

dic["knn"] =  accuracy_score(y_val, knn.predict(x_val))

# decision tree
 
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(max_depth = 4, random_state = seed)
tree.fit(x_train_2, y_train_2)

dic["decision_tree"] = accuracy_score(y_val, tree.predict(x_val))

# random forest

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators= 100, random_state = seed)
rf.fit(x_train_2, y_train_2)

dic["random_forest"] = accuracy_score(y_val, rf.predict(x_val))

# linear support vector machine

from sklearn.svm import LinearSVC
svc = LinearSVC()

svc.fit(x_train_2, y_train_2)
dic["svc"] = accuracy_score(y_val, svc.predict(x_val))

# evaluating models

validation = pd.Series(dic, name="algorithms accuracy")
print() ; print(validation)

# training final model

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)  

rf.fit(x_train, y_train)

score = str(accuracy_score(y_test, lr.predict(x_test)))

print() ; print("final accuracy: ", score)