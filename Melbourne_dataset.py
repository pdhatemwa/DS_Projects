import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

Melbourne_data_path = "C:/Users/patri/Downloads/melb_data.csv.zip"

# Attempt to view the data

My_data = pd.read_csv(Melbourne_data_path)

# To view the first 5 rows of the data, use the ".head()" command

# print(My_data.describe())

# print(My_data.columns)
# print('------HOUSING PRICES------')
# print(f"{My_data['Price']}")
y = My_data['Price']

# Create the list of features below
feature_names = ['Suburb', 'Address', 'Rooms',
        'Landsize', 'BuildingArea', 'YearBuilt'
        , 'Regionname']
# Select data corresponding to features in feature_names
X = My_data[feature_names]
# print(X)
# print(y)

# Since the data is looking good, it's time to go into some data cleaning.
# We want to eliminate data rows and columns with NaN values.

combined = pd.concat([X, y], axis = 1).dropna() # Concatenate columns not rows
# To concatenate rows, we would use axis = 0
# pd.concat combines features (X) and (y) side by side into a single data frame.

X_clean = combined[feature_names]
y_clean = combined['Price']

# Use one-hot encode for categorical variables.
X_encoded = pd.get_dummies(X_clean)
# One-hot encoding is a technique used to convert 
# categorical data (text labels) into a numeric format 
# that machine learning algorithms can work with.
#It creates a new binary column for each category, marking:
# 1 if the original value matches that category
# 0 otherwise.

# Split the data into training data and test data.
X_train, X_valid, y_train, y_valid = train_test_split(X_encoded, y_clean, random_state=1)
# Random state = 1, means


# Now it is time to fit the model 
# so that we can eventually predict prices.

model = RandomForestRegressor(random_state = 1)
model.fit(X_train, y_train)

# After fitting the model, we then go ahead to evaluate it
# And this can be done using the mean absolute error from statistics.

preds = model.predict(X_valid)
mae = mean_absolute_error(y_valid, preds)
print("Mean Absolute Error:", mae)
print('\n')
# Mean Absolute Error (MAE) measures the 
# average absolute difference between actual values (y_valid) and predicted values (preds). 
# It tells you how far off your model’s predictions are on average.
print("---- SUMMARY STATISTICS OF CLEANED y-values-----")
print(y_clean.describe())
print('\n')
print("-----M.A.E as percentage of mean.----")
my_mae_ratio = mae/y_clean.mean()
my_percentage_mae = my_mae_ratio * 100
print(f"Percentage mean absolute error: {my_percentage_mae:.2f} %")
print('\n')



