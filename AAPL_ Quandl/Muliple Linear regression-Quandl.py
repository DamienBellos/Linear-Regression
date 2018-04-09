import quandl
import math
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing, model_selection, svm
import matplotlib.pyplot as plt
from matplotlib import style
import datetime
import pickle

style.use('ggplot')

quandl.ApiConfig.api_key = '*place your Quandl key here*'
df = quandl.get('EOD/AAPL')
df = df.drop(['Open', 'High', 'Low', 'Close', 'Volume', 'Dividend', 'Split'], axis=1)

# print (df.head())
df['HL_PCT'] = (df['Adj_High'] - df['Adj_Low']) / df['Adj_Close'] * 100
df['PCT_change'] = (df['Adj_Close'] - df['Adj_Open']) / df['Adj_Open'] * 100

df = df[['Adj_Close', 'HL_PCT', 'PCT_change', 'Adj_Volume']]
forecast_col = 'Adj_Close'
df.fillna(value=-9999, inplace=True)
forecast_out = int(math.ceil(0.01 * len(df)))

df['label'] = df[forecast_col].shift(-forecast_out)

X = np.array(df.drop(['label'], 1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

df.dropna(inplace=True)

y = np.array(df['label'])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)
lreg = LinearRegression()

# using already create pickle file for the regression
# lreg.fit(X_train, y_train)
# confidence = lreg.score(X_test, y_test)
# print(confidence)

# with open('mLinearRegression.pickle', 'wb') as f:
#     pickle.dump(lreg, f)

pickle_in = open('mLinearRegression.pickle', 'rb')
lreg = pickle.load(pickle_in)

forecast_set = lreg.predict(X_lately)
df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += 86400
    df.loc[next_date] = [np.nan for _ in range(len(df.columns) - 1)] + [i]

df['Adj_Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
