from sklearn.datasets import load_diabetes
import pandas as pd

# Load the diabetes dataset
diabetes = load_diabetes()

# Create a DataFrame
df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
df['target'] = diabetes.target

# Save to CSV
df.to_csv('diabetes_data.csv', index=False)
print("Diabetes data saved as diabetes_data.csv")