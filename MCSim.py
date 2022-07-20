# importing dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
import datetime as dt


# import data
def get_data(stocks, start, end):
    stockData = pdr.get_data_yahoo(stocks, start, end)
    stockData = stockData['Close']
    returns = stockData.pct_change()
    meanReturns = returns.mean()
    covMatrix = returns.cov()
    return meanReturns, covMatrix


# stocks in portfolio
stockList = ['AAPL', 'TSLA', 'GME', 'AMC', 'F', 'BAC']

# processing the stock data
stocks = [stock for stock in stockList]

endDate = dt.datetime.now()
startDate = endDate - dt.timedelta(days=300)

meanReturns, covMatrix = get_data(stocks, startDate, endDate)

# weighting the stocks
weights = np.random.random(len(meanReturns)) #If you want to specify your weights, you could make an array that has the same length as the number of stocks but just manually enter the weights. Just make sure the weights add up to 1. IE: weights=[.2, .2, .2, .1, .1, .2]
weights /= np.sum(weights)

# Monte Carlo Method

mc_sims = 100  # number of simulations
T = 365  # timeframe in days

meanM = np.full(shape=(T, len(weights)), fill_value=meanReturns)
meanM = meanM.T

portfolio_sims = np.full(shape=(T, mc_sims), fill_value=0.0)

initialPortfolioValue = 10000

L = np.linalg.cholesky(covMatrix)

for m in range(0, mc_sims):
    Z = np.random.normal(size=(T, len(weights)))
    dailyReturns = meanM + np.inner(L, Z)
    portfolio_sims[:, m] = np.cumprod(np.inner(weights, dailyReturns.T)+1) * initialPortfolioValue

plt.plot(portfolio_sims)
plt.ylabel('Portfolio Value ($)')
plt.xlabel('Days')
plt.title('Monte Carlo Simulation of Portfolio')
plt.hlines(y=initialPortfolioValue, xmin=0, xmax=T, linewidth=2, color='black')
plt.show()

