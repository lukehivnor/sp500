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
    min_length = min(len(dow), len(sp), len(nasdaq), len(vtsax))


    dow = dow.iloc[:min_length]  # value return
    sp = sp.iloc[:min_length]  # value return
    nasdaq = nasdaq.iloc[:min_length]  # value return
    vtsax = vtsax.iloc[:min_length]  # Date Open
    # Combine the separate DataFrames into a list
    stocks = [dow, sp, nasdaq, vtsax]



    # Create a DateTime index starting from 2011-01-01 with daily frequency
    dates = pd.date_range(start='2011-12-05', periods=len(sp), freq='D')
    print(len(sp))
    print(len(dow))
    print(len(nasdaq))
    print(len(vtsax))
    # Set the index to be the DateTimeIndex
    dow.index = dates
    sp.index = dates
    nasdaq.index = dates
    vtsax.index = dates

    # Combine all stock returns into a single DataFrame
    df = pd.concat((i for i in stocks), axis=1)

    # Now resample the data to weekly frequency
    df_weekly = df.resample('W').apply(lambda x: (1 + x).prod() - 1)

    df = df_weekly

    return df


df = getter()
df[df <= 0] = np.nan


# Calculate log returns

percentage_returns = df.pct_change()
log_returns = np.log(1 + percentage_returns)
log_returns = log_returns.dropna()
mean_returns = log_returns.mean()  # Average returns for each stock
cov_matrix = log_returns.cov()     # Covariance matrix for the stocks
# Define Sharpe ratio objective function (maximize)




def negative_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate=0.01):
    portfolio_return = np.dot(weights, mean_returns)
    portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return -(portfolio_return - risk_free_rate) / portfolio_risk  # Maximize Sharpe ratio

# Initial guess for weights (equal distribution)
initial_weights = np.array([1 / 4] * 4)

# Constraints: sum of weights = 1
constraints = [{'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}]

# Bounds for weights: between 0 and 1 (no short selling)
bounds = [(-0.5, 0.5) for _ in range(4)]

# Perform optimization
result = minimize(negative_sharpe_ratio, initial_weights, args=(mean_returns, cov_matrix, 0.01),
                  method='SLSQP', bounds=bounds, constraints=constraints)

# Extract results
optimal_weights = result.x
optimal_return = np.dot(optimal_weights, mean_returns)
optimal_risk = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))
optimal_sharpe = (optimal_return - 0.01) / optimal_risk

# Output
print("Optimal Weights:", optimal_weights)
print("Portfolio Return:", optimal_return)
print("Portfolio Risk (Std Dev):", optimal_risk)
print("Sharpe Ratio:", optimal_sharpe)