import numpy as np
import pandas as pd
from scipy.optimize import minimize


def getter():
    dower = pd.read_csv("D://tester//dow_10_yr.csv")
    sper = pd.read_csv("D://tester//sp500_10_yr.csv")
    nasdaqer = pd.read_csv("D://tester//nasdaq_comp_10_yr.csv")
    vtsax_flipped = pd.read_csv("D://tester//vtsax_10_yr.csv")
    vtsaxer = vtsax_flipped.iloc[::-1]

    dow = dower.drop(' value', axis=1)
    dow = dow.drop('date', axis=1)
    sp = sper.drop(' value', axis=1)
    sp = sp.drop('date', axis=1)
    nasdaq = nasdaqer.drop(' value', axis=1)
    nasdaq = nasdaq.drop('date', axis=1)
    vtsax = vtsaxer.drop('Date', axis=1)
    vtsax = vtsax.drop(' Open', axis=1)
    print(dow)  # value return
    print(sp)  # value return
    print(nasdaq)  # value return
    print(vtsax)  # Date Open
    # Combine the separate DataFrames into a list
    stocks = [dow, sp, nasdaq, vtsax]
    df = pd.concat((i for i in stocks), axis=1)
    return df


df = getter()
# Calculate mean returns and covariance matrix
mean_returns = df.mean()  # Average returns for each stock
cov_matrix = df.cov()     # Covariance matrix for the stocks
risk_free_rate = 0.036  # Example annual risk-free rate (1%)

# Define the objective function to maximize the Sharpe Ratio
def negative_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
    portfolio_return = np.dot(weights, mean_returns)  # Portfolio return
    portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))  # Portfolio risk (std deviation)
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_risk  # Sharpe ratio
    return -sharpe_ratio  # Negate to maximize Sharpe ratio

# Constraint: The sum of the weights must be 1
constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})

# Initial guess for weights (equal distribution)
initial_weights = np.array([1/4] * 4)

# Bounds for weights: between 0 and 1
bounds = [(0.025, 0.75) for _ in range(4)]

# Perform optimization
result = minimize(negative_sharpe_ratio, initial_weights,
                  args=(mean_returns, cov_matrix, risk_free_rate),
                  method='SLSQP', bounds=bounds, constraints=constraints)

# Optimal weights
optimal_weights = result.x

# Portfolio metrics
optimal_return = np.dot(optimal_weights, mean_returns)
optimal_risk = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))
optimal_sharpe_ratio = (optimal_return - risk_free_rate) / optimal_risk

# Output results
print("Optimal Weights:", optimal_weights)
print("Portfolio Return:", optimal_return)
print("Portfolio Risk (Std Dev):", optimal_risk)
print("Sharpe Ratio:", optimal_sharpe_ratio)
