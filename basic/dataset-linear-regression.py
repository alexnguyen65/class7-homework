# Perform linear regression and print performance.
# Correlation between the number of rooms and the price of the house.
#
# Ref. https://medium.com/@amitg0161/sklearn-linear-regression-tutorial-with-boston-house-dataset-cde74afd460a

import numpy as np
import pandas as pd
from sklearn.datasets import load_boston

boston_housing = load_boston()
boston_df = pd.DataFrame(boston_housing.data, columns=boston_housing.feature_names)
boston_df['PRICE'] = boston_housing.target

X_rooms = boston_df.RM
y_price = boston_df.PRICE

X_rooms = np.array(X_rooms).reshape(-1,1)
y_price = np.array(y_price).reshape(-1,1)

# Splitting features and target datasets into: train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_rooms, y_price, test_size=0.30)

# Training a Linear Regression model with fit()
print("Training linear regression for RM vs PRICE...")
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train, y_train)

# Predicting the results for our test dataset
predicted_values = lm.predict(X_train)

# Calculating and printing the model performance
# Ref. https://towardsdatascience.com/how-to-select-the-right-evaluation-metric-for-machine-learning-models-part-1-regrression-metrics-3606e25beae0
#
from sklearn.metrics import mean_squared_error
rmse = (np.sqrt(mean_squared_error(y_train, predicted_values)))
r2 = round(lm.score(X_train, y_train),2)

print("The model performance for training set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
print("\n")

