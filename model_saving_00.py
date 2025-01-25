import joblib

# Save the trained model
best_model = models['Random Forest'].best_estimator_  # or any final model
joblib.dump(best_model, 'house_price_predictor.joblib')

# Later, load the saved model
loaded_model = joblib.load('house_price_predictor.joblib')

# Use the model to make predictions
new_data = [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]]  # Example new input
predictions = loaded_model.predict(new_data)
print(predictions)
