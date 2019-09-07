# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 00:45:24 2019

@author: darre
"""


import pandas as pd
import numpy as np
import math
#to plot within notebook
from sklearn import preprocessing
import matplotlib.pyplot as plt


from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from datetime import *
#split our data
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
%matplotlib inline
#read the file
df = pd.read_csv('AAPL.csv')

#print the head
#df.head()
df.tail()

close_px = df['Adj Close']
mavg = close_px.rolling(window=100).mean()

#Return Deviation
rets = close_px / close_px.shift(1) - 1

#Prediciting stock
dfreg = df.loc[:,['Adj Close','Volume']]
dfreg['HL_PCT'] = (df['High'] - df['Low']) / df['Close'] * 100.0
dfreg['PCT_change'] = (df['Close'] - df['Open']) / df['Open'] * 100.0


#Preprocessing and prediction
# Drop missing value
dfreg.fillna(value=-99999, inplace=True)
# We want to separate 1 percent of the data to forecast
forecast_out = int(math.ceil(0.01 * len(dfreg)))
# Separating the label here, we want to predict the AdjClose
forecast_col = 'Adj Close'
dfreg['label'] = dfreg[forecast_col].shift(-forecast_out)
X = np.array(dfreg.drop(['label'], 1))
# Scale the X so that everyone can have the same distribution for linear regression
X = preprocessing.scale(X)
# Finally We want to find Data Series of late X and early X (train) for model generation and evaluation
X_lately = X[-forecast_out:]
X = X[:-forecast_out]
# Separate label and identify it as y
y = np.array(dfreg['label'])
y = y[:-forecast_out]

#from sklearn.cross_validation import train_test_spilt
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2)

# Linear regression
clfreg = LinearRegression(n_jobs=-1)
clfreg.fit(X_train, y_train)
# Quadratic Regression 2
clfpoly2 = make_pipeline(PolynomialFeatures(2), Ridge())
clfpoly2.fit(X_train, y_train)

# Quadratic Regression 3
clfpoly3 = make_pipeline(PolynomialFeatures(3), Ridge())
clfpoly3.fit(X_train, y_train)

# KNN Regression
clfknn = KNeighborsRegressor(n_neighbors=2)
clfknn.fit(X_train, y_train)


confidencereg = clfreg.score(X_test, y_test)
confidencepoly2 = clfpoly2.score(X_test,y_test)
confidencepoly3 = clfpoly3.score(X_test,y_test)
confidenceknn = clfknn.score(X_test, y_test)

print("Value of each model is")
print("inear regression : | ", confidencereg)
print("Quadratic Regression 2 : | ", confidencepoly2)
print("Quadratic Regression 3: | ", confidencepoly3)
print(" KNN Regression: | ", confidenceknn)

forecast_set = clfreg.predict(X_lately)
dfreg['Forecast'] = np.nan

last_date = dfreg.iloc[-1].name
last_unix = last_date
next_unix = last_unix

for i in forecast_set:
    next_date = next_unix
    next_unix += 1
    dfreg.loc[next_date] = [np.nan for _ in range(len(dfreg.columns)-1)]+[i]
    print("Using Linear regression our forcast for next" ,next_unix - last_unix , " days is the following: " ,[i] )

dfreg['Adj Close'].tail(500).plot()

dfreg['Forecast'].tail(500).plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()


