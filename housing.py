import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("Housing.csv")  # Replace with your actual dataset
print(df.head())  # Check first few rows

# Preprocess data (handling missing values, encoding if necessary)
df = df.dropna()  # Simple approach to remove missing values

# Select features (X) and target (y)
X = df[['area']]  
y = df['price']  

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"R² Score: {r2}")

# Plot regression line
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred, color='red', label='Regression Line')
plt.xlabel("Size")
plt.ylabel("Price")
plt.legend()
plt.title("Linear Regression Model")
plt.show()

# Interpretation
print(f"Intercept: {model.intercept_}")
print(f"Coefficient: {model.coef_[0]} - Indicates change in price per unit increase in size.")
