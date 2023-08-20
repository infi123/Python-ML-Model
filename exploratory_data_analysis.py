import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
data = pd.read_csv('train.csv')

# Display the first few rows of the dataframe
print(data.head())

# Get the summary statistics of the numerical variables
print(data.describe())

# Visualize the distribution of the target variable 'SalePrice'
sns.histplot(data['SalePrice'], kde=True)
plt.show()

# Data preprocessing: Separate numeric and categorical columns
numeric_cols = data.select_dtypes(include='number')
categorical_cols = data.select_dtypes(include='object')

# Perform one-hot encoding on categorical columns
categorical_cols_encoded = pd.get_dummies(categorical_cols, drop_first=True)

# Concatenate numeric and encoded categorical data
data_processed = pd.concat([numeric_cols, categorical_cols_encoded], axis=1)

# Correlation matrix
corr_matrix = data_processed.corr()

# Visualize the correlation matrix using a heatmap
plt.figure(figsize=(12, 12))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()
