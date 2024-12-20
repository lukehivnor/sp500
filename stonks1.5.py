import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import numpy as np
from datetime import datetime
import joblib
from xgboost import plot_importance
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import RandomizedSearchCV


def getter():
    dower = pd.read_csv("D://tester//dow_10_yr.csv")
    sper = pd.read_csv("D://tester//sp500_10_yr.csv")
    nasdaqer = pd.read_csv("D://tester//nasdaq_comp_10_yr.csv")
    vtsax_flipped = pd.read_csv("D://tester//vtsax_10_yr.csv")
    vtsaxer = vtsax_flipped.iloc[::-1]

    dow = dower.drop(' value', axis=1)
    dow = dow.drop('date', axis=1)
    sp = sper.drop(' value', axis=1)
    #sp = sp.drop('date', axis=1)
    nasdaq = nasdaqer.drop(' value', axis=1)
    nasdaq = nasdaq.drop('date', axis=1)
    vtsax = vtsaxer.drop('Date', axis=1)
    vtsax = vtsax.drop(' Open', axis=1)
    min_length = min(len(dow), len(sp), len(nasdaq), len(vtsax))



    # Create a DateTime index starting from 2011-01-01 with daily frequency
    dates = pd.date_range(start='2011-12-05', periods=len(sp), freq='D')
    dates = range(3294)
    print(len(sp))
    print(len(dow))
    print(len(nasdaq))
    print(len(vtsax))

    # Set the index to be the DateTimeIndex
    #dow.index = dates
    sp['date'] = dates
    sp.fillna(sp['return'].mean())

    sp['ones'] = 1
    #nasdaq.index = dates
    #vtsax.index = dates
    stocks = [dow, sp, nasdaq, vtsax]

    return stocks


stonks = getter()

df = stonks[1]


# Split the dataset
train_size = int(0.7 * len(df))
val_size = int(0.2 * len(df))

train = df.iloc[:train_size]
validation = df.iloc[train_size:train_size + val_size]
test = df.iloc[train_size + val_size:]

'''
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
}

'''

X_train = pd.DataFrame(train['date'].values, columns=['return']).fillna(train['return'].mean())
y_train = pd.Series(train['return'].values).fillna(train['return'].mean())  # 1D Series

# For validation and test sets, reshape appropriately:
X_val = pd.DataFrame(validation['date'].values, columns=['return']).fillna(train['return'].mean())  # 2D DataFrame
y_val = pd.Series(validation['return'].values).fillna(train['return'].mean())  # 1D Series

X_test = pd.DataFrame(test['date'].values, columns=['return']).fillna(train['return'].mean())  # 2D DataFrame
y_test = pd.Series(test['return'].values).fillna(train['return'].mean())  # 1D Series

X_train_np = X_train.values
X_val_np = X_val.values
X_test_np = X_test.values

# Create DMatrix with numerical data
dtrain = xgb.DMatrix(X_train_np, label=y_train)
dval = xgb.DMatrix(X_val_np, label=y_val)


# Define parameters
params = {
    'objective': 'reg:squarederror',  # Regression problem
    'learning_rate': 0.01,            # Small learning rate
    'max_depth': 7,                   # Depth of each tree
    'min_child_weight': 10,           # Minimum instance weight in a child
    'subsample': 0.25,                # Fraction of data used for each tree
    'colsample_bytree': 0.7,         # Fraction of features used per tree
    'eval_metric': 'rmse',           # Root Mean Squared Error metric
}
# Add learning rate decay (reducing rate during training)


# Train using xgb.train with early stopping
evals = [(dtrain, 'train'), (dval, 'validation')]
model = xgb.train(
    params,
    dtrain,
    num_boost_round=1000,
    evals=evals,
    early_stopping_rounds=20
)


# Predict on the test set
dtest = xgb.DMatrix(X_test_np)  # Convert test data to DMatrix as well
y_pred = model.predict(dtest)

# Check for NaN values in predictions or actuals and fill if necessary
y_test_series = pd.Series(y_test)
y_pred_series = pd.Series(y_pred)

y_test_series_filled = y_test_series.fillna(y_test_series.mean())
y_pred_series_filled = y_pred_series.fillna(y_pred_series.mean())

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test_series_filled, y_pred_series_filled))
print(f"Test RMSE: {rmse}")

# Predict performance
predicted_performance = model.predict(xgb.DMatrix(3299))
print(f"Predicted Performance: {predicted_performance[0]}")

'''
param_grid = {
    'max_depth': [3, 5, 7, 15],
    'min_child_weight': [1, 5, 10],
    'subsample': [0.25, 0.6, 1.0],
    'colsample_bytree': [0.3, 0.5, 0.7],
    'learning_rate': [0.01, 0.07, 0.15],
    'n_estimators': [100, 200, 500]
}

# Randomized search to find best hyperparameters
xgb_model = xgb.XGBRegressor()
randomized_search = RandomizedSearchCV(xgb_model, param_grid, n_iter=20, cv=4, verbose=2, random_state=42)
randomized_search.fit(X_train, y_train)
best_params = randomized_search.best_params_
print(f"Best Hyperparameters: {best_params}")
'''