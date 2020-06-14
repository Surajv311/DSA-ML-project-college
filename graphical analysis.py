"""THE FOLLOWING CODE IS FOR THE GRAHICAL ANALYSIS(LR AND SVM MODEL) FOR THE GIVEN DATASET """
"""
THE FOLLOWING M.L PROJECT WOULD PREDICT THE ESTIMATED VALUE OF STOCKS OF A COMPANY IN FUTURE """
#Importing libraries
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

"""COMPANY : RELIANCE INDUSTRIES """

#Loading dataset
data_Stocks= pd.read_csv("reliance company data reverse.csv")
#print(data_Stocks.head(5)) # would print the first 5 values of dataset
#print(data_Stocks.tail(5))#would print last 5 values of dataset
#Create the list; X and Y data sets
dates_S = []
prices_S = []
#Get the number of rows and columns in the data set
#print(data_Stocks.shape)
#Store all of the data except for the last row in our variable > we are revising the varaible's value
data_Stocks = data_Stocks.head(len(data_Stocks)-1)
#print(data_Stocks)
#print(data_Stocks.shape) # to confirm that the n-1 data has been stored by checking shape
#Get all of the rows from the Date Column
data_Dates = data_Stocks.loc[:, 'Date']
#Get all of the rows from the Open Column
data_Price = data_Stocks.loc[:, 'Open Price']
# Create the independent data set X
for date in data_Dates:
    dates_S.append([int(date.split('-')[1])])
# we have split it in different months
# Create the dependent data set Y
for open_price in data_Price:
    prices_S.append(float(open_price))

#See what days/months/years were recorded
#print(dates_S)

def predict_prices(dates_S, prices_S, x):
    # Create the 3 Support Vector Regression models
    svr_lin = SVR(kernel='linear', C=1e3)
    svr_poly = SVR(kernel='poly', C=1e3, degree=2)
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1) # KERNEL METHODS ARE CLASS OF ALGORITHMS FOR PATTERN ANALYSIS

    # Train the SVR models
    svr_lin.fit(dates_S, prices_S)
    svr_poly.fit(dates_S, prices_S)
    svr_rbf.fit(dates_S, prices_S)

    # Create the Linear Regression model
    lin_reg = LinearRegression()
    # Train the Linear Regression model
    lin_reg.fit(dates_S, prices_S)

    # Plot the models on a graph to see which has the best fit
    plt.scatter(dates_S, prices_S, color='black', label='Data')
   # plt.plot(dates_S, svr_rbf.predict(dates_S), color='orange', label='SVR RBF')
    #plt.plot(dates_S, svr_poly.predict(dates_S), color='green', label='SVR Poly')
    plt.plot(dates_S, svr_lin.predict(dates_S), color='blue', label='SVR Linear')
    plt.plot(dates_S, lin_reg.predict(dates_S), color='red', label='Linear Reg')
    plt.xlabel('Months_STOCKS')
    plt.ylabel('Price_STOCKS')
    plt.title('Regression MODEL')
    plt.legend()
    plt.show()
    return svr_rbf.predict(x)[0], svr_lin.predict(x)[0], svr_poly.predict(x)[0], lin_reg.predict(x)[0]
#Predict the price of Reliance on day 28
predicted_price = predict_prices(dates_S, prices_S, [[12]]) # we have considered month 12 now instead of day ...
#print(predicted_price)
# note : in case of overfitting the model may be great at training data but not good for new data
