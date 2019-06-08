# Perform k-nearest neighbors regression and print performance.
# Correlation between the number of rooms and the price of the house.

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

# Training a Regression model based on k-nearest neighbors
print("Training regression based on k-nearest neighbors for RM vs PRICE...")
from sklearn.neighbors import KNeighborsRegressor
for n in range (2,6):
   lm = KNeighborsRegressor(n_neighbors=n)
   lm.fit(X_train, y_train)

   # Predicting the results for our test dataset
   predicted_values = lm.predict(X_train)

   # Calculating and printing the model performance
   from sklearn.metrics import mean_squared_error
   rmse = (np.sqrt(mean_squared_error(y_train, predicted_values)))
   r2 = round(lm.score(X_train, y_train),2)

   print("The model performance for training set")
   print("--------------------------------------")
   print("n_neighbors=", n)
   print('RMSE is {}'.format(rmse))
   print('R2 score is {}'.format(r2))
   print("\n")

