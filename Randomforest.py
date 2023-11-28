import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

data = pd.read_csv('weather_data.csv')
data['date'] = pd.to_datetime(data[['year', 'month', 'day']])

X = data[['temperature_night', 'year', 'month']]
y = data['temperature_day']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# hyperparameters ranges
estimators = [100, 150, 200]
max_depth = [10, 15, 20]
min_samples_split = [2, 4, 6]
min_samples_leaf = [1, 2, 4]

best_mse = float('inf')
best_params = {}

# Iterative process for tuning
for n_estimators in estimators:
    for depth in max_depth:
        for split in min_samples_split:
            for leaf in min_samples_leaf:
                # Train the model with the current given hyperparameters
                model = RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=depth,
                    min_samples_split=split,
                    min_samples_leaf=leaf,
                    random_state=42
                )
                model.fit(X_train_scaled, y_train)

                #evaluation
                test_predictions = model.predict(X_test_scaled)
                mse = mean_squared_error(y_test, test_predictions)

                # MSE comparisons to keep the best hyperparameters
                if mse < best_mse:
                    best_mse = mse
                    best_params = {
                        'n_estimators': n_estimators,
                        'max_depth': depth, # Updated to use new iterator variable
                        'min_samples_split': split, #--->---#
                        'min_samples_leaf': leaf #--->---#
                    }

print(f'Best MSE: {best_mse}')
print(f'Best Parameters: {best_params}')

# Train the final model with the new best parameters
final_model = RandomForestRegressor(
    n_estimators=best_params['n_estimators'],
    max_depth=best_params['max_depth'],
    min_samples_split=best_params['min_samples_split'],
    min_samples_leaf=best_params['min_samples_leaf'],
    random_state=42
)
final_model.fit(X_train_scaled, y_train)

# Final evaluation
final_predictions = final_model.predict(X_test_scaled)
final_mse = mean_squared_error(y_test, final_predictions)
print(f'Final Model MSE: {final_mse}')
