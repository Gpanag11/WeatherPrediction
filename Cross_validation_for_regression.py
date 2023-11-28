import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import numpy as np

data = pd.read_csv('weather_data.csv')
data['date'] = pd.to_datetime(data[['year', 'month', 'day']])

#features and target variable
X = data[['temperature_night', 'year', 'month']]
y = data['temperature_day']

# Create a pipeline that first scales the data and then applies linear regression
model = make_pipeline(StandardScaler(), LinearRegression())

# Apply 10-fold cross-validation
scores = cross_val_score(model, X, y, cv=10, scoring='neg_mean_squared_error')

# Convert scores to positive values, because they are returned as negative by default
mse_scores = -scores

#average mse calculation
average_mse = np.mean(mse_scores)
print(f'Average MSE from 10-fold cross-validation: {average_mse}')