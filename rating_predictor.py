import pandas as pd
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
# First, remove any rows where the target 'Rating' column is NaN
df = df.dropna(subset=['Rating'])

# Impute missing values in feature columns with the mean
imputer = SimpleImputer(strategy='mean')
df[['RatingDist1', 'RatingDist2', 'RatingDist3', 'RatingDist4', 'RatingDist5', 'CountsOfReview']] = imputer.fit_transform(df[['RatingDist1', 'RatingDist2', 'RatingDist3', 'RatingDist4', 'RatingDist5', 'CountsOfReview']])

# Ensure the feature columns are numeric (in case of any string values)
df[['RatingDist1', 'RatingDist2', 'RatingDist3', 'RatingDist4', 'RatingDist5', 'CountsOfReview']] = df[['RatingDist1', 'RatingDist2', 'RatingDist3', 'RatingDist4', 'RatingDist5', 'CountsOfReview']].apply(pd.to_numeric, errors='coerce')

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
prediction = model.predict([user_input])

print(f"Predicted Rating: {prediction[0]:.2f}")
print(f"R2 Score: {r2:.2f}")
print(f"Mean Squared Error: {mse:.2f}")
