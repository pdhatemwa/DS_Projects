import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

Melbourne_data_path = "C:/Users/patri/Downloads/melb_data.csv.zip"

# Attempt to view the data

My_data = pd.read_csv(Melbourne_data_path)

# To view the first 5 rows of the data, use the ".head()" command

print(My_data.describe())

print(My_data.columns)
print('------HOUSING PRICES------')
print(f"{My_data['Price']}")
y = My_data['Price']

# Create the list of features below
feature_names = ['Suburb', 'Address', 'Rooms',
        'Landsize', 'BuildingArea', 'YearBuilt'
        , 'Regionname']
# Select data corresponding to features in feature_names
X = My_data[feature_names]
print(X)
print(y)

# Since the data is looking good, it's time to go into some data cleaning.
# We want to eliminate
