import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


# Функція для отримання введених даних від користувача
def get_user_input():
    print("Введіть розподіл оцінок для кожної категорії (від 1 до 5) та кількість відгуків, а також мову:")
    rating_dist1 = float(input("Введіть розподіл оцінки 1 (кількість відгуків з оцінкою 1): "))
    rating_dist2 = float(input("Введіть розподіл оцінки 2 (кількість відгуків з оцінкою 2): "))
    rating_dist3 = float(input("Введіть розподіл оцінки 3 (кількість відгуків з оцінкою 3): "))
    rating_dist4 = float(input("Введіть розподіл оцінки 4 (кількість відгуків з оцінкою 4): "))
    rating_dist5 = float(input("Введіть розподіл оцінки 5 (кількість відгуків з оцінкою 5): "))
    counts_of_review = int(input("Введіть загальну кількість відгуків: "))
    language = input("Введіть мову (наприклад, 'eng'): ")
    return [rating_dist1, rating_dist2, rating_dist3, rating_dist4, rating_dist5, counts_of_review, language]


# Завантаження даних (для тренування моделі)
df = pd.read_csv('CSV_BI_Lab1_data_source.csv', sep=';')

# Видалення ком у колонці 'Rating' та перетворення її на тип float
df['Rating'] = df['Rating'].replace({',': '.'}, regex=True).astype(float)

# Обробка відсутніх значень
df = df.dropna(subset=['Rating'])

# Заповнення відсутніх значень у колонках ознак середнім значенням
imputer = SimpleImputer(strategy='mean')
df[['RatingDist1', 'RatingDist2', 'RatingDist3', 'RatingDist4', 'RatingDist5',
    'CountsOfReview']] = imputer.fit_transform(
    df[['RatingDist1', 'RatingDist2', 'RatingDist3', 'RatingDist4', 'RatingDist5', 'CountsOfReview']])

# Переконатися, що всі ознаки мають числові значення (на випадок наявності строкових значень)
df[['RatingDist1', 'RatingDist2', 'RatingDist3', 'RatingDist4', 'RatingDist5', 'CountsOfReview']] = df[
    ['RatingDist1', 'RatingDist2', 'RatingDist3', 'RatingDist4', 'RatingDist5', 'CountsOfReview']].apply(pd.to_numeric,
                                                                                                         errors='coerce')

# Кодування колонки 'Language' за допомогою OneHotEncoder
encoder = OneHotEncoder(sparse_output=False)  # Виправлення: використовуємо sparse_output=False замість sparse=False
language_encoded = encoder.fit_transform(df[['Language']])
language_encoded_df = pd.DataFrame(language_encoded, columns=encoder.get_feature_names_out(['Language']))

# Додавання закодованої мови до DataFrame
df = pd.concat([df, language_encoded_df], axis=1)

# Вибір ознак для тренування моделі (включаючи закодовану мову)
X = df[['RatingDist1', 'RatingDist2', 'RatingDist3', 'RatingDist4', 'RatingDist5', 'CountsOfReview'] + list(
    language_encoded_df.columns)]
y = df['Rating']

# Переконатися, що немає NaN значень в X та y
X = X.dropna()
y = y[X.index]  # Узгоджуємо y з ознаками (після видалення NaN з X)

# Розподіл на тренувальні та тестові дані
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Тренування моделі лінійної регресії
model = LinearRegression()
model.fit(X_train, y_train)

# Оцінка моделі
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

# Отримання введених даних від користувача та передбачення рейтингу
user_input = get_user_input()

# Перетворення введених даних користувача в DataFrame з відповідними назвами колонок
user_input_data = user_input[:-1]  # Видалення мови з введених даних для one-hot кодування
user_input_language = user_input[-1]  # Витягуємо мову окремо

# Перетворення введених даних користувача у DataFrame для ознак
user_input_df = pd.DataFrame([user_input_data], columns=X.columns[:-len(language_encoded_df.columns)])

# One-hot кодування введеної мови
user_input_language_encoded = encoder.transform([[user_input_language]])
user_input_language_df = pd.DataFrame(user_input_language_encoded, columns=encoder.get_feature_names_out(['Language']))

# Об'єднуємо введені дані з закодованою мовою
user_input_final = pd.concat([user_input_df, user_input_language_df], axis=1)

# Прогнозуємо за допомогою моделі
prediction = model.predict(user_input_final)

# Виведення результатів
print(f"Прогнозований рейтинг: {prediction[0]:.2f}")
print(f"R2 Score: {r2:.2f}")
print(f"Mean Squared Error: {mse:.2f}")

# Візуалізація

# 1. Фактичні vs Прогнозовані рейтинги
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')
plt.title('Фактичні vs Прогнозовані Рейтинги')
plt.xlabel('Фактичні Рейтинги')
plt.ylabel('Прогнозовані Рейтинги')
plt.show()

# 2. Важливість ознак (за допомогою коефіцієнтів)
plt.figure(figsize=(8, 6))
features = X.columns
importance = model.coef_
plt.barh(features, importance, color='green')
plt.title('Важливість Ознак (Коефіцієнти Лінійної Регресії)')
plt.xlabel('Значення коефіцієнтів')
plt.ylabel('Ознаки')
plt.show()

# 3. Розподіл рейтингів
plt.figure(figsize=(8, 6))
plt.hist(y, bins=20, color='orange', edgecolor='black')
plt.title('Розподіл Рейтингів')
plt.xlabel('Рейтинг')
plt.ylabel('Частота')
plt.show()
