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
# Calculate mean returns and covariance matrix
mean_returns = df.mean()  # Average returns for each stock
cov_matrix = df.cov()     # Covariance matrix for the stocks
target_return = 0.036  # Target return (e.g., 1%)

# Define the objective function: Minimize portfolio risk (standard deviation)
def portfolio_risk(weights, cov_matrix):
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

# Constraints
constraints = [
    {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1},  # Weights sum to 1
    {'type': 'ineq', 'fun': lambda weights: np.dot(weights, mean_returns) - target_return}  # Ensure target return
]

# Initial guess for weights (equal distribution)
initial_weights = np.array([1 / 4] * 4)

# Bounds for weights: between 0 and 1
bounds = [(0, 1) for _ in range(4)]

# Perform optimization
result = minimize(portfolio_risk, initial_weights, args=(cov_matrix,),
                  method='SLSQP', bounds=bounds, constraints=constraints)

# Optimal weights
optimal_weights = result.x

# Portfolio metrics
optimal_return = np.dot(optimal_weights, mean_returns)
optimal_risk = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))

# Output results
print("Optimal Weights:", optimal_weights)
print("Portfolio Return:", optimal_return)
print("Portfolio Risk (Std Dev):", optimal_risk)