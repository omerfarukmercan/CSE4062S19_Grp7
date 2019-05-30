from dateutil.parser import parse
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
plt.rcParams.update({'figure.figsize': (10, 7), 'figure.dpi': 120})
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from pandas.plotting import register_matplotlib_converters
import statsmodels
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.ar_model import AR
from pandas.plotting import lag_plot
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from pandas import Series
from pandas import Series
from matplotlib import pyplot
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.ar_model import AR

#AUTO REGRESSION


result = pd.read_excel('data2.xlsx')
result['date'] = pd.to_datetime(result['date'])
data = result.loc[:, ['value']]
data = data.set_index(result.date)
data['value'] = pd.to_numeric(data['value'], downcast='float', errors='coerce')

X = data.values

plot_acf(data, lags=50)
plt.show()

result2 = pd.read_excel('test2.xlsx')
result2['date'] = pd.to_datetime(result2['date'])
data2 = result2.loc[:, ['value']]
data2 = data.set_index(result.date)
data2['value'] = pd.to_numeric(data['value'], downcast='float', errors='coerce')

X2 = data2.values

"""
series = Series.from_csv('daily-minimum-temperatures.csv', header=0)
# split dataset
X = series.values
train, test = X[1:len(X)-7], X[len(X)-7:]
"""
train, test = X, X2



model = AR(train)
model_fit = model.fit()
window = model_fit.k_ar
coef = model_fit.params
# walk forward over time steps in test
history = train[len(train)-window:]
history = [history[i] for i in range(len(history))]
predictions = list()

for t in range(len(test)):
    length = len(history)
    lag = [history[i] for i in range(length-window,length)]
    yhat = coef[0]

    for d in range(window):
        yhat += coef[d+1] * lag[window-d-1]

    obs = test[t]
    predictions.append(yhat)
    history.append(obs)
    print('predicted=%f, expected=%f' % (yhat, obs))


error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)

error2 = mean_absolute_error(test, predictions)
print('Test MAE: %.3f' % error2)

error3 = r2_score(test, predictions)
print('Test R2: %.3f' % error3)

# plot
pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()

