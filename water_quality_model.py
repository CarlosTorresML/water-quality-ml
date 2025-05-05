# --------------------------------------
#            Pre-processing
# --------------------------------------
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from time import time

# Load the dataset into a DF
dataset = pd.read_csv('water_potability.csv')

# Checking amount of NaN
print('Total amount of NaN:', dataset.isna().sum().sum())
print('\nNaN per column:\n', dataset.isna().sum())
print('\nChecking mean and median:\n',dataset[['Sulfate', 'Trihalomethanes']].describe())
# Mean and median are similar in both cases so we are going to use mean
# For the ph I use the median since the column may have extreme values

# Imputing values
median_ph = dataset['ph'].median()
dataset['ph'] = dataset['ph'].fillna(median_ph)

mean_sulfate = dataset['Sulfate'].mean()
dataset['Sulfate'] = dataset['Sulfate'].fillna(mean_sulfate)

mean_thlmt = dataset['Trihalomethanes'].mean()
dataset['Trihalomethanes'] = dataset['Trihalomethanes'].fillna(mean_thlmt)

# Checking NaN values again
print('Total amount of NaN after preprocessing:', dataset.isna().sum().sum())

# Round ph values to a single decimal
dataset['ph'] = dataset['ph'].round(1)

# Round all columns to a 3 decimals except potability
cols_to_round = dataset.columns.drop('Potability')
dataset[cols_to_round] = dataset[cols_to_round].round(3)


# --------------------------------------
#          Classification Model
# --------------------------------------

# Classification split
X_class = dataset.drop(columns='Potability')
y_class = dataset['Potability']
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X_class, y_class, random_state=1, test_size=0.10)

# Create the classificator model and training
rf_class = RandomForestClassifier(random_state=1)
rf_class.fit(X_train_class, y_train_class)

# Predict and evaluate
y_pred_test_class = rf_class.predict(X_test_class)
y_pred_train_class = rf_class.predict(X_train_class)
print(f'\nAccuracy of the RF Classificator in training: {accuracy_score(y_train_class, y_pred_train_class)}')
print(f'Accuracy of the RF Classificator in test: {accuracy_score(y_test_class, y_pred_test_class)}\n')
# We can se that the forest is overfitted

# Tuning random forest
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [11, 13, 14],
    "min_samples_leaf": [1]
}

# Creating the GridSearchCV object
cv = GridSearchCV(estimator=rf_class,
                  param_grid=param_grid,
                  verbose=2)
start = time()
cv.fit(X_train_class, y_train_class)
stop = time()
print(f'\nBest parameters: {cv.best_params_}')
print(f'It took {stop-start:.2f} seconds.')
# Create the new model with the optimal parameters
rf_class_optimized = RandomForestClassifier(random_state=1,
                                            class_weight='balanced',
                                            n_estimators=cv.best_params_['n_estimators'],
                                            max_depth=cv.best_params_['max_depth'])

# Training the model
rf_class_optimized.fit(X_train_class, y_train_class)
y_pred_test_class_optimized = rf_class_optimized.predict(X_test_class)

# Comparing results
print(f'\nNon optimized model accuracy: {accuracy_score(y_test_class, y_pred_test_class)}')
print(f'Optimized model accuracy: {accuracy_score(y_test_class, y_pred_test_class_optimized)}')


# --------------------------------------
#            Regression Model
# --------------------------------------

# Regression split
X_reg = dataset.drop(columns='ph')
y_reg = dataset['ph']
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, random_state=1, test_size=0.20)

# Create and train the model
reg_model = GradientBoostingRegressor(random_state=1)
reg_model.fit(X_train_reg, y_train_reg)

# Predict and evaluate
y_pred_test_reg = reg_model.predict(X_test_reg)
print(f'\nRMSE in Gradient Boosting Regressor: {np.sqrt(mean_squared_error(y_test_reg, y_pred_test_reg))}')


print()
print('*** Summary ***')
print(f'\tAccuracy of the RF Classificator in training: {accuracy_score(y_train_class, y_pred_train_class)}')
print(f'\tAccuracy of the RF Classificator in test: {accuracy_score(y_test_class, y_pred_test_class)}')
print(f'\tRMSE in Gradient Boosting Regressor: {np.sqrt(mean_squared_error(y_test_reg, y_pred_test_reg))}')


# --------------------------------------
#              Conclusions
# --------------------------------------

# - The default Random Forest model was clearly overfitted, showing perfect accuracy on training data and poor generalisation.
# - Multiple hyperparameter tuning attempts failed to significantly reduce overfitting or improve accuracy.
# - Applying class_weight='balanced' led to a small but real improvement in test accuracy.
# - For regression, Gradient Boosting outperformed previous Random Forest results with a lower RMSE.
# - These results show that tuning does not always outperform defaults, and dataset limitations may cap performance.
