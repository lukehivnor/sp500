import numpy as np
import pandas as pd
from scipy.optimize import minimize

def get_df():

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

    # Calculate the mean returns and the covariance matrix
    mean_returns = df.mean()  # Average return for each stock
    cov_matrix = df.cov()  # Covariance matrix for the stocks

    # Define the objective function: Minimize risk (std deviation) and maximize return
    def objective(weights, mean_returns, cov_matrix):
        portfolio_return = np.dot(weights, mean_returns)  # Expected return
        portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))  # Portfolio standard deviation (risk)

        # Negative of the return to maximize it, add the risk as part of the objective
        return portfolio_risk - portfolio_return

    # Constraint: The sum of the weights should be 1
    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})

    # Initial guess for weights (equal distribution)
    initial_weights = np.array([1 / len(stocks)] * len(stocks))

    # Bounds for each weight to be between 0 and 1
    bounds = [(0.1, 0.5) for _ in range(len(stocks))]

    # Perform the optimization
    result = minimize(objective, initial_weights, args=(mean_returns, cov_matrix),
                      method='SLSQP', bounds=bounds, constraints=constraints)

    # Optimal weights
    optimal_weights = result.x

    # Output the results
    print("Optimal Weights:", optimal_weights)
    print("Minimized Risk and Maximized Return:", result.fun)





get_df()

'''
# Constraint: The sum of the weights should be 1
constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})

# Initial guess for weights (equal distribution)
initial_weights = np.array([1 / len(df.columns)] * len(df.columns))

# Bounds for each weight to be between 0 and 1
bounds = [(0, 1) for _ in range(len(df.columns))]

# Perform the optimization
result = minimize(objective, initial_weights, args=(mean_returns, cov_matrix),
                  method='SLSQP', bounds=bounds, constraints=constraints)

# Optimal weights
optimal_weights = result.x

print("Optimal Weights:", optimal_weights)
print("Minimized Risk and Maximized Return:", result.fun)

'''