import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle

# Load the diabetes data
df = pd.read_csv('diabetes_data.csv')
X = df.drop('target', axis=1)
y = df['target']

# Split our data into practice and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Teach our computer about diabetes
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Test how well it learned
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"Our computer's Mean Squared Error: {mse:.2f}")
print(f"Our computer's R² Score: {r2:.2f}")
print("Model trained successfully!")

# Save what it learned
with open('diabetes_model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model saved as diabetes_model.pkl!")