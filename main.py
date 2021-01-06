#!/usr/bin/env python3

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

seed = 666

data = pd.read_csv('heart_failure.csv')
print(data.head())

x = data.iloc[:, 0:-1].values # features: from first to next-to-last column
y = data.iloc[:, -1].values # label: last column

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 1/3, random_state = seed)

x_train_2, x_val, y_train_2, y_val = train_test_split(
    x_train, y_train, test_size = 1/3, random_state = seed)

# feature scaling
scaler = MinMaxScaler()
x_train_2 = scaler.fit_transform(x_train_2)
x_val = scaler.transform(x_val)

dic = {} # result dictionary

def logistic_regression(): 
    lr = LogisticRegression(random_state = seed)
    lr.fit(x_train_2, y_train_2)

    return accuracy_score(y_val, lr.predict(x_val))
    
def knn():
    knn = KNeighborsClassifier(n_neighbors = 7)
    knn.fit(x_train_2, y_train_2)

    return accuracy_score(y_val, knn.predict(x_val))

def decision_tree(): 
    tree = DecisionTreeClassifier(max_depth = 4, random_state = seed)
    tree.fit(x_train_2, y_train_2)

    return accuracy_score(y_val, tree.predict(x_val))

def random_forest():
    rf = RandomForestClassifier(
        n_estimators= 100, random_state = seed)
    rf.fit(x_train_2, y_train_2)

    return accuracy_score(y_val, rf.predict(x_val))

def linear_support_vector_machine():
    svc = LinearSVC()

    svc.fit(x_train_2, y_train_2)
    return accuracy_score(y_val, svc.predict(x_val))

# evaluating models
dic["logistic_regression"] = logistic_regression()
dic["knn"] = knn()
dic["decision_tree"] = decision_tree()
dic["random_forest"] = random_forest()
dic["svc"] = linear_support_vector_machine()

validation = pd.Series(dic, name="algorithms accuracy")
print() ; print(validation)

# train final model
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)  

rf = RandomForestClassifier(
    n_estimators= 100, random_state = seed)
rf.fit(x_train, y_train)

score = str(accuracy_score(y_test, rf.predict(x_test)))
print() ; print("final accuracy: ", score)