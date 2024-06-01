import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV

#Convert data to Pandas DataFrame
rental = pd.read_csv(r"C:\Users\akesli\Documents\GitHub\Data Camp Codes\Tree Based Model\rental_info.csv")

#Convert rental_date and return_date to Pandas datatime
rental['rental_date'] = pd.to_datetime(rental['rental_date'])
rental['return_date'] = pd.to_datetime(rental['return_date'])

#Calculate the how many dates a dvd has been rented by a customer
rental['rental_length_days'] = (rental['return_date'] - rental['rental_date']).dt.days

#Create dummy variables for deleted scenes and behind the scenes
rental['deleted_scenes'] = np.where(rental['special_features'].str.contains('Deleted Scenes'), 1, 0)
rental['behind_the_scenes'] = np.where(rental['special_features'].str.contains('Behind the Scenes'), 1, 0)

#Seperate target column from others
cols_to_drop = ['special_features', 'rental_length_days', 'rental_date', 'return_date']
X = rental.drop(cols_to_drop, axis=1)
y = rental['rental_length_days']

#Seperate the train and test data as 20% test and 80% train random_state=9
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=9)

#import lasso and instantiate to data
from sklearn.linear_model import Lasso
rental_lasso = Lasso(random_state=9, alpha=0.1)
rental_lasso_coef = rental_lasso.fit(X, y).coef_
X_lasso_train, X_lasso_test = X_train.iloc[:, rental_lasso_coef >0], X_test.iloc[:, rental_lasso_coef > 0]

#Import OLS models and run on data
from sklearn.linear_model import LinearRegression
ols = LinearRegression()
ols = ols.fit(X_lasso_train, y_train)
y_test_pred = ols.predict(X_lasso_test)
mse_lin_reg_lasso = mean_squared_error(y_test, y_test_pred)

#Import random forest regressor and apply to data
from sklearn.ensemble import RandomForestRegressor
param_dist = {'n_estimators' : np.arange(1, 101, 1),
             'max_depth' : np.arange(1,11,1)}
rf = RandomForestRegressor()
rand_search = RandomizedSearchCV(rf, param_distributions=param_dist, cv=5, random_state=9)
rand_search.fit(X_train, y_train)
hyper_params = rand_search.best_params_
rf = RandomForestRegressor(n_estimators=hyper_params['n_estimators'],
                          max_depth=hyper_params['max_depth'],
                          random_state=9)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
mse_random_forest = mean_squared_error(y_test, rf_pred)

print(mse_lin_reg_lasso)
print(mse_random_forest)

best_model = rf
best_mse = mse_random_forest