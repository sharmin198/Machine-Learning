# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 12:58:14 2018

@author: Sharmin
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing datasets

dataset = pd.read_csv("Salary_Data.csv")
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 1].values

#Splitting the dataset into the training set and test set

from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1/3, random_state = 0)

#Fitting Simple Linear Regression to the Training set

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

#Predicting the test set result

Y_pred = regressor.predict(X_test)

#Visualising the training set results

plt.scatter(X_train, Y_train, color = "red")
plt.plot(X_train, regressor.predict(X_train), color = "blue")
plt.title("Salary vs Experience(Training set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

#visualising the test set results

plt.scatter(X_test, Y_test, color = "red")
plt.plot(X_train, regressor.predict(X_train), color = "blue")
plt.title("Salary vs Experience(Test set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()



