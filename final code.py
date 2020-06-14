"""THE FOLLOWING M.L PROJECT WOULD PREDICT THE ESTIMATED VALUE OF STOCKS OF A COMPANY IN FUTURE """

import quandl #this package is used to get the finnacial data
import pandas as pd # for data analysis tools
import numpy as np # has array processing package for scientific computing
from sklearn.linear_model import LinearRegression # to perform the linear regression
from sklearn.svm import SVR # TO USE SVM ALGORITHM
from sklearn.model_selection import train_test_split # to train and split our data
import matplotlib.pyplot as plt # to plot our graph

#Loading the data_Stocks from .csv

"""COMPANY : RELIANCE INDUSTRIES """

#data_Stocks is our data frame
data_Stocks= pd.read_csv("reliance company data reverse.csv") #getting the data and storing it in data_Stocks
print("<FIRST VALUE FROM DATA SET>")
print(data_Stocks.head(1)) # TO DISPLAY FIRST VALUE FROM THE  data_Stocks SET
print("\n")
print("<LAST VALUE FROM DATA SET>")
print(data_Stocks.tail(1)) # TO DISPLAY LAST  VALUES FROM THE  data_Stocks SET
#Now we are we are revising the data in our variable data_Stocks as shown
data_Stocks = data_Stocks[['Open Price']]
#TO DISPLAY the newly revised data
print(data_Stocks.head(1)) # TO DISPLAY THE FIRST VALUE
#now we declare a variable to predict the prices out in for 'd' days in future
predict_Price = 30 #initialising the value to 30 as we will predict for 30 days
print("\n\n")
#now we create another column (the target or dependent variable) shifted 'u' units up
"""
This would shift the data by '1' and the last place would be NULL.Hence by this logic we would store the predicted value in the new coloumn
data_Stocks['Prediction_Price'] =data_Stocks[['Open Price']].shift(-1) # shifts only one row from the 'Open Price' and puts the remaining set into 'Prediction_Price'
print(data_Stocks.head())
print(data_Stocks.tail())
#we can notice that the last value is null as we have shifted and copied the rows into 'Prediction_Price' from 'Open Price'
Now we use the above logic
"""
data_Stocks['Prediction_Price'] =data_Stocks[['Open Price']].shift(-predict_Price) # since predict_Price = 30 , top 30 would be removed and rest shifted up
#print(data_Stocks.head())
#print(data_Stocks.tail()) # hence we can notice the top 30 have been removed then the rows are shifted up .So the last 30 value is NULL

"""Now we create the independent data set (X)"""

# Convert the dataframe data_Stocks to a numpy array
#Also we would drop the 'Prediction_Price' coloumn as we would now be using independent variable
X = np.array(data_Stocks.drop(['Prediction_Price'],1)) #numpy array is just like lists in python but more efficient
#Remove the last '30' rows i.e predict_Price
X = X[:-predict_Price]
#print(X)


"""Now we create the dependent data set (Y)"""


# we Convert the dataframe data_Stocks to a numpy array (All of the values including the NaN's)
Y= np.array(data_Stocks['Prediction_Price']) # as we don't need the 'Open Price' coloumn now
# Get all of the Y values except the last 30 rows
Y = Y[:-predict_Price]
#print(Y)
# Now split the data into x% training and y% testing in test_size , i.e = 0.8 ( 80% training 20% testing)
# more training data -> better model , more testing data -> high accuracy on testing results
x_train, x_test, y_train, y_test = train_test_split(X , Y, test_size=0.8)
# Now using SVM (Regressor)
s_Vector_Reg = SVR(kernel='rbf', C=1e3, gamma=0.1) #C is regularization parameter and gamma is a parameter that defines influence of single training
s_Vector_Reg.fit(x_train, y_train) # gamma is influence , smaller gamma ~ high influence ~ highly constrained model

# Testing Model: 'Score' would return the coefficient of determination R^2 of the prediction.
#best score = 1 , coefficient of determination = square of corelation between 'x' and 'y' scores
svm_Confidence_Value = s_Vector_Reg.score(x_test, y_test)
print("svm confidence value : ", svm_Confidence_Value)
# we create the L.regression model
linear_Regression = LinearRegression()
# now we train the model
linear_Regression .fit(x_train, y_train)
#confidence score is how much a predicted base can be trusted
linear_Regression_Confidence_Value = linear_Regression.score(x_test, y_test)
print("linear regression confidence value : ", linear_Regression_Confidence_Value)
# HENCE LINEAR REFRESSION MODEL IS BETTER THAN SVM MODEL

# Set x_Predict_ equal to the last 30 rows of the original data set from 'Open Price' column
x_Predict_ = np.array(data_Stocks.drop(['Prediction_Price'],1))[-predict_Price:]
#print(x_Predict_) # this is the data that we are going to do prediction on
print("\n")
# Linear regression model prediction for  next 30 days
linear_Reg_Prediction = linear_Regression.predict(x_Predict_)
print("LINEAR REGRESSION PREDICTION:")
print("\n")
print(linear_Reg_Prediction)

print("\n")

# Support vector regressor model prediction for the next 30 days
svm_Reg_Prediction = s_Vector_Reg.predict(x_Predict_)
print("SVM REGRESSION PREDICTION:")
print("\n")
print(svm_Reg_Prediction)
