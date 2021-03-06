#!/usr/bin/env python3

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from utils import println

seed = 666

data = pd.read_csv('heart_failure.csv')
println(data.head())

X = data.iloc[:, 0:-1].values # features: from first to next-to-last column
y = data.iloc[:, -1].values # label: last column

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 1/3, random_state = seed)

X_train_2, X_val, y_train_2, y_val = train_test_split(
    X_train, y_train, test_size = 1/3, random_state = seed)

# feature scaling
scaler = MinMaxScaler()
X_train_2 = scaler.fit_transform(X_train_2)
X_val = scaler.transform(X_val)

dic = {} # result dictionary

def logistic_regression(): 
    lr = LogisticRegression(random_state = seed)
    lr.fit(X_train_2, y_train_2)
    return accuracy_score(y_val, lr.predict(X_val))
    
def knn():
    knn = KNeighborsClassifier(n_neighbors = 7)
    knn.fit(X_train_2, y_train_2)
    return accuracy_score(y_val, knn.predict(X_val))

def decision_tree(): 
    tree = DecisionTreeClassifier(max_depth = 4, random_state = seed)
    tree.fit(X_train_2, y_train_2)
    return accuracy_score(y_val, tree.predict(X_val))

def random_forest():
    rf = RandomForestClassifier(
        n_estimators= 100, random_state = seed)
    rf.fit(X_train_2, y_train_2)
    return accuracy_score(y_val, rf.predict(X_val))

def linear_support_vector_machine():
    svc = LinearSVC()
    svc.fit(X_train_2, y_train_2)
    return accuracy_score(y_val, svc.predict(X_val))

# evaluating models
dic["logistic_regression"] = logistic_regression()
dic["knn"] = knn()
dic["decision_tree"] = decision_tree()
dic["random_forest"] = random_forest()
dic["svc"] = linear_support_vector_machine()

validation = pd.Series(dic, name="algorithms accuracy")
println(validation)

# train final model
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)  

rf = RandomForestClassifier(
    n_estimators= 100, random_state = seed)
rf.fit(X_train, y_train)

score = str(accuracy_score(y_test, rf.predict(X_test)))
print("final accuracy:", score)