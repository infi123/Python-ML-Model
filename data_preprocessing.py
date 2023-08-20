import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv('train.csv')

# List of numerical columns
numerical_cols = ['LotArea', 'OverallQual', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

# List of categorical columns
categorical_cols = ['HouseStyle', 'Heating', 'CentralAir', 'SaleCondition', 'SaleType']

# Separate the data into numerical and categorical dataframes
data_numerical = data[numerical_cols]
data_categorical = data[categorical_cols]

# Perform standard scaling on the numerical columns
scaler = StandardScaler()
data_numerical = pd.DataFrame(scaler.fit_transform(data_numerical), columns=data_numerical.columns)

# Perform one-hot encoding on categorical columns
encoder = OneHotEncoder(drop='first')
data_categorical_encoded = encoder.fit_transform(data_categorical)

# Convert the sparse matrix to a dense numpy array
data_categorical_encoded = data_categorical_encoded.toarray()

# Concatenate numerical and categorical data
data_preprocessed = np.concatenate([data_numerical, data_categorical_encoded], axis=1)

# Separate the target variable
y = data['SalePrice']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(data_preprocessed, y, test_size=0.2, random_state=42)

# Save the preprocessed data
np.savez('data_preprocessed.npz', X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
