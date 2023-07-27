# Transform the test set with the selected features
X_test_selected = rfecv.transform(X_test)

# Make predictions on the test set
y_pred = model.predict(X_test_selected)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# Print evaluation metrics
print("Mean Squared Error (MSE):", mse)
print("Mean Absolute Error (MAE):", mae)
# Plot actual vs. predicted prices
plt.scatter(y_test, y_pred)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs. Predicted Prices')
plt.show()

from sklearn.metrics import r2_score
print("R2 score is ",r2_score(y_true=y_test, y_pred=y_pred),"\n")

print("An R-squared score of 0.92 indicates that approximately 92% of the variance in the target variable is explained by the model. In other words, the model captures a large portion of the variation in the target variable and provides a good fit to the data.\n")
