#this program is to optimize a user's portfolio using the Efficient Frontier from Financial Economics Theroy

#import libraries
from pandas_datareader import data as web
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
plt.style.use('fiverthirdtyeight')

#get the stock symbols/ tickers in the portfolio 
assets = ['XEI.TO','ZDY.TO','VSP.TO','ZQQ.TO']

#Assign weights to stocks. Weights = 1 
weights = np.array([0.25,0.25,0.25,0.25])

# Get the stocks/portfolio starting date
stockStartDate ='2013-01-01'

#Get the stocks ending date (today)
today = datetime.today().strftime('%Y-%m-%d')

#create a dataframe to store the adjusted close price of the stocks 
df = pd.DataFrame()

#store the adjusted close price to the dataframe
for stock in assets: 
    df[stock]=web.DataReader(stock,'yahoo',start=stockStartDate, end=today)['Adj Close']

#show dataframe
df

#Visually show the stock /portfolio 
title = 'Portfolio Adj. Close Price History'

#Get the stocks 
my_stocks = df

#Create and plot graph
for c in my_stocks.columns.values:
    plt.plot(my_stocks[c],label=c)

plt.title(title)
plt.xlabel('Date', fontsize = 18)
plt.ylabel('Adj Close Price CAD', fontsize = 18)
plt.legend(my_stocks.columns.values,loc ='upper left')
plt.show()

#show daily simple return
returns = df.pct_change()
#(new price - old price)/old price
returns



#create the annualized covariance matrix
cov_matrix_annual = returns.cov()*252
cov_matrix_annual
#square root of the diagonals give you volatility

#calculate the portfolio variance
port_variance = np.dot(weights.T,np.dot(cov_matrix_annual,weights))
port_variance

# Calculate the portfolio volatility aka standard deviation
port_volatility = np.sqrt(port_variance)
print(port_volatility)

#Calculate the annual portfolio return, by 252 trading days
portfolioSimpleAnnualReturn = np.sum(returns.mean()*weights*252)
portfolioSimpleAnnualReturn

#show the expected annual returns, volatility (risk), and variance 
percent_var = str(round(port_variance,2)*100)+'%'
percent_volatility = str(round(port_volatility,2)*100)+'%'
percent_return = str(round(portfolioSimpleAnnualReturn,2)*100)+'%'

print('Expected annual return: '+ percent_return)
print('annual volatility/risk: '+ percent_volatility)
print('Annual variance: '+ percent_var)

from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models 
from pypfopt import expected_returns

#Portfolio Optimization

#calculate expected returns and annualized sample covariance matrix of asset returns
mu = expected_returns.mean_historical_return(df)
#sample covariance matrix
S = risk_models.sample_cov(df)

#optimize for max Sharpe ratio, how much excess return you receive per standard deviation
ef = EfficientFrontier(mu, S)
weights = ef.max_sharpe()
#set weights to be some absolute cut offs 
cleaned_weights = ef.clean_weights()
print(cleaned_weights)
ef.portfolio_performance(verbose=True)

#get discrete allocation of each share per stock
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

latest_prices = get_latest_prices(df)
weights = cleaned_weights
da = DiscreteAllocation(weights, latest_prices, total_portfolio_value = 40000)

allocation, leftover = da.lp_portfolio()
print('Discrete allocation',allocation)
print('Funds remaining: ${:.2}'.format(leftover))