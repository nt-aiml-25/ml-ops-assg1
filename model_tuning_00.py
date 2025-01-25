from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.datasets import fetch_california_housing
import pandas as pd

# Load the dataset
data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(),
}

# Hyperparameter tuning (example for Random Forest)
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, None]
}
rf_search = GridSearchCV(RandomForestRegressor(), param_grid, cv=3)
models['Random Forest'] = rf_search

# Train and evaluate models
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f'{name} Mean Squared Error: {mse}')
    
import joblib

# Save the trained model
best_model = models['Random Forest'].best_estimator_  # or any final model
joblib.dump(best_model, 'house_price_predictor.joblib')

# Later, load the saved model
loaded_model = joblib.load('house_price_predictor.joblib')

# Use the model to make predictions
#new_data = [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]]  # Example new input
new_data = [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]]  # Example new input
predictions = loaded_model.predict(new_data)
print(predictions)
