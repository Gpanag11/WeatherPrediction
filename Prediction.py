import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from LinearRegression import LinearRegression
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('weather_data.csv')
# data.info()

# data.head()

data['date'] = pd.to_datetime(data[['year', 'month', 'day']])
data.set_index('date', inplace=True)

plt.figure(figsize=(15, 5))

plt.figure(figsize=(12, 5))

# Create plots for temperature_day and temperature_night
plt.subplot(1, 2, 1)
sns.histplot(data['temperature_day'], kde=True, bins=30)
plt.title('Day Temperature Distribution')

plt.subplot(1, 2, 2)
sns.histplot(data['temperature_night'], kde=True, bins=30)
plt.title('Night Temperature Distribution')

plt.tight_layout()
plt.show()

plt.figure(figsize=(15, 10))

# Scatter plot for temperature_day vs year
plt.subplot(2, 2, 1)
sns.scatterplot(x='year', y='temperature_day', data=data)
plt.title('Day Temperature Yearly')

# Scatter plot for temperature_day vs month
plt.subplot(2, 2, 2)
sns.scatterplot(x='month', y='temperature_day', data=data)
plt.title('Day Temperature mountly')

# Adjust layout and show the plots
plt.tight_layout()
plt.show()

# reset the dataframe to its original form
data = data.reset_index(drop=True)
print(data.head())

# Split the dataset into training and testing sets based on the date
training_data = data.loc[data['year'] <= 2020]
testing_data = data.loc[data['year'] > 2020]

# Define X and y for training data
X_train = training_data[['temperature_night', 'year', 'month']]
y_train = training_data['temperature_day']

# Define X and y for testing data
X_test = testing_data[['temperature_night', 'year', 'month']]
y_test = testing_data['temperature_day']

# Show the shape of the training and test sets
print((X_train.shape, X_test.shape, y_train.shape, y_test.shape))

#apply standard scaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

reg = LinearRegression(lr=0.01)
reg.fit(X_train_scaled, y_train)
predictions = reg.predict(X_test_scaled)


def mse(y_test, predictions):
    return np.mean((y_test - predictions) ** 2)


mse_value = mse(y_test, predictions)
print(f'Mean Squared Error is: {mse_value}')

plt.figure(figsize=(10, 6))
plt.scatter(y_test, predictions, alpha=0.6)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--k', linewidth=2)
plt.xlabel('Actual Temperature')
plt.ylabel('Predicted Temperature')
plt.title('Actual vs. Predicted Temperatures')
plt.show()
