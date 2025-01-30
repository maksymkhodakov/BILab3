import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer


# Function to get user input
def get_user_input():
    print("Введіть розподіл оцінок для кожної категорії (від 1 до 5) та кількість відгуків:")
    rating_dist1 = float(input("Enter Rating Distribution 1 (кількість відгуків з оцінкою 1): "))
    rating_dist2 = float(input("Enter Rating Distribution 2 (кількість відгуків з оцінкою 2): "))
    rating_dist3 = float(input("Enter Rating Distribution 3 (кількість відгуків з оцінкою 3): "))
    rating_dist4 = float(input("Enter Rating Distribution 4 (кількість відгуків з оцінкою 4): "))
    rating_dist5 = float(input("Enter Rating Distribution 5 (кількість відгуків з оцінкою 5): "))
    counts_of_review = int(input("Enter Counts of Review (загальна кількість відгуків): "))
    return [rating_dist1, rating_dist2, rating_dist3, rating_dist4, rating_dist5, counts_of_review]


# Load data (for model training)
df = pd.read_csv('CSV_BI_Lab1_data_source.csv', sep=';')

# Remove commas in 'Rating' and convert it to float
df['Rating'] = df['Rating'].replace({',': '.'}, regex=True).astype(float)

# Handle missing values
df = df.dropna(subset=['Rating'])

# Impute missing values in feature columns with the mean
imputer = SimpleImputer(strategy='mean')
df[['RatingDist1', 'RatingDist2', 'RatingDist3', 'RatingDist4', 'RatingDist5',
    'CountsOfReview']] = imputer.fit_transform(
    df[['RatingDist1', 'RatingDist2', 'RatingDist3', 'RatingDist4', 'RatingDist5', 'CountsOfReview']])

# Ensure the feature columns are numeric (in case of any string values)
df[['RatingDist1', 'RatingDist2', 'RatingDist3', 'RatingDist4', 'RatingDist5', 'CountsOfReview']] = df[
    ['RatingDist1', 'RatingDist2', 'RatingDist3', 'RatingDist4', 'RatingDist5', 'CountsOfReview']].apply(pd.to_numeric,
                                                                                                         errors='coerce')

# Feature selection for training the model
X = df[['RatingDist1', 'RatingDist2', 'RatingDist3', 'RatingDist4', 'RatingDist5', 'CountsOfReview']]
y = df['Rating']

# Ensure there are no NaN values in both X and y
X = X.dropna()
y = y[X.index]  # Align y with the features (after dropping NaNs from X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

# Get user input and predict rating
user_input = get_user_input()

# Convert the user input into a DataFrame with the correct column names
user_input_df = pd.DataFrame([user_input], columns=X.columns)

# Predict using the model
prediction = model.predict(user_input_df)

# Display results
print(f"Predicted Rating: {prediction[0]:.2f}")
print(f"R2 Score: {r2:.2f}")
print(f"Mean Squared Error: {mse:.2f}")

# Visualizations

# 1. Actual vs Predicted Ratings
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')
plt.title('Actual vs Predicted Ratings')
plt.xlabel('Actual Ratings')
plt.ylabel('Predicted Ratings')
plt.show()

# 2. Feature Importance (using coefficients)
plt.figure(figsize=(8, 6))
features = X.columns
importance = model.coef_
plt.barh(features, importance, color='green')
plt.title('Feature Importance (Linear Regression Coefficients)')
plt.xlabel('Coefficient Value')
plt.ylabel('Features')
plt.show()

# 3. Distribution of Ratings
plt.figure(figsize=(8, 6))
plt.hist(y, bins=20, color='orange', edgecolor='black')
plt.title('Distribution of Ratings')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.show()
