import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import string


# --- Допоміжні функції ---

def extract_letter_features(title):
    """
    Для заданої назви книги повертаємо словник із ознаками:
    для кожної літери англійського алфавіту (a-z) 1, якщо вона зустрічається в назві, або 0, якщо ні.
    """
    title = str(title).lower()
    features = {}
    for letter in string.ascii_lowercase:
        features[f'letter_{letter}'] = 1 if letter in title else 0
    return features


def get_user_input():
    """
    Отримує введення користувача для прогнозування рейтингу книги.
    Повертає: рік публікації, мову, назву книги та ознаки з назви.
    """
    print("\nВведіть дані для прогнозування рейтингу книги:")
    publish_year = int(input("Введіть рік публікації: "))
    language = input("Введіть мову (наприклад, 'eng'): ")
    title = input("Введіть назву книги: ")
    # Отримуємо ознаки з назви
    letter_features = extract_letter_features(title)
    return publish_year, language, title, letter_features


# --- Завантаження та підготовка даних ---

# Завантаження даних з CSV; роздільник ';'
df = pd.read_csv('CSV_BI_Lab1_data_source.csv', sep=';')

# Прибираємо зайві пробіли в назвах колонок
df.columns = [col.strip() for col in df.columns]

# Замінюємо кому на крапку в колонці 'Rating' та переводимо у тип float
df['Rating'] = df['Rating'].replace({',': '.'}, regex=True).astype(float)

# Видаляємо записи з пропущеним рейтингом
df = df.dropna(subset=['Rating'])

# Створення ознак з назви книги (Name)
letter_features_df = df['Name'].apply(extract_letter_features).apply(pd.Series)

# Об'єднуємо оригінальний DataFrame з ознаками з назви
df = pd.concat([df, letter_features_df], axis=1)

# Підготовка числових ознак: PublishYear
features_numeric = ['PublishYear']
df[features_numeric] = df[features_numeric].apply(pd.to_numeric, errors='coerce')
imputer = SimpleImputer(strategy='mean')
df[features_numeric] = imputer.fit_transform(df[features_numeric])

# One-hot кодування для колонки 'Language'
encoder = OneHotEncoder(sparse_output=False)
language_encoded = encoder.fit_transform(df[['Language']])
language_encoded_df = pd.DataFrame(language_encoded, columns=encoder.get_feature_names_out(['Language']))
df = pd.concat([df, language_encoded_df], axis=1)

# Остаточний набір ознак: PublishYear + закодована мова + ознаки з назви (letter_a ... letter_z)
feature_cols = features_numeric + list(language_encoded_df.columns) + [col for col in df.columns if
                                                                       col.startswith('letter_')]
X = df[feature_cols]
y = df['Rating']

# Видаляємо можливі NaN (якщо вони є)
X = X.dropna()
y = y[X.index]

# Розподіл даних на тренувальну та тестову вибірки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Тренування моделі ---
model = LinearRegression()
model.fit(X_train, y_train)

# Оцінка моделі
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print(f"Linear Regression: R2 Score = {r2:.2f}, Mean Squared Error = {mse:.2f}")

# --- Business Value Explanation ---
print("\nBusiness Value:")
print("Ця модель прогнозує рейтинг книги на основі наступних чинників:")
print(" - Рік публікації: дозволяє врахувати тренди та зміни вподобань аудиторії з часом.")
print(" - Мова: різні ринки можуть мати різне сприйняття книг.")
print(" - Літери в назві: аналізує, чи впливає стиль оформлення назви (наявність певних літер) на рейтинг.")
print("\nВидавці можуть використовувати цей аналіз для оптимізації назв книг, адаптуючи їх під цільову аудиторію,")
print("а також для прогнозування потенційної успішності книги перед її виходом на ринок.\n")

# --- Перший прогноз ---
print("=== Перший прогноз ===")
publish_year, language, title, user_letter_features = get_user_input()

# Формуємо DataFrame для введених даних
user_data = pd.DataFrame({
    'PublishYear': [publish_year]
})

# One-hot кодування введеної мови (використовуємо вже навчений encoder)
user_language_encoded = encoder.transform([[language]])
user_language_df = pd.DataFrame(user_language_encoded, columns=encoder.get_feature_names_out(['Language']))

# Перетворення ознак з назви (вже отриманий словник) у DataFrame
user_letter_df = pd.DataFrame([user_letter_features])

# Об'єднуємо введені дані з усіма необхідними ознаками
user_input_final = pd.concat([user_data, user_language_df, user_letter_df], axis=1)

# Переконуємося, що всі колонки присутні у потрібному порядку
for col in X.columns:
    if col not in user_input_final.columns:
        user_input_final[col] = 0
user_input_final = user_input_final[X.columns]

# Прогноз для першої книги
predicted_rating = model.predict(user_input_final)[0]
print(f"\nПрогнозований рейтинг книги: {predicted_rating:.2f}")

# Перевірка, чи існує книга в дата-сеті (порівнюємо назву, ігноруючи регістр)
mask = df['Name'].str.lower() == title.lower()
if mask.any():
    real_rating = df.loc[mask, 'Rating'].iloc[0]
    print(f"Реальний рейтинг книги: {real_rating:.2f}")
else:
    print("Книга не знайдена в дата-сеті, порівняння неможливе.")

# --- Додаткові прогнози ---
while True:
    choice = input("\nБажаєте зробити ще один прогноз? (y/n): ").strip().lower()
    if choice != 'y':
        print("Завершення програми.")
        break

    publish_year, language, title, user_letter_features = get_user_input()

    user_data = pd.DataFrame({
        'PublishYear': [publish_year]
    })
    user_language_encoded = encoder.transform([[language]])
    user_language_df = pd.DataFrame(user_language_encoded, columns=encoder.get_feature_names_out(['Language']))
    user_letter_df = pd.DataFrame([user_letter_features])
    user_input_final = pd.concat([user_data, user_language_df, user_letter_df], axis=1)
    for col in X.columns:
        if col not in user_input_final.columns:
            user_input_final[col] = 0
    user_input_final = user_input_final[X.columns]

    predicted_rating = model.predict(user_input_final)[0]
    print(f"\nПрогнозований рейтинг книги: {predicted_rating:.2f}")

    mask = df['Name'].str.lower() == title.lower()
    if mask.any():
        real_rating = df.loc[mask, 'Rating'].iloc[0]
        print(f"Реальний рейтинг книги: {real_rating:.2f}")
    else:
        print("Книга не знайдена в дата-сеті, порівняння неможливе.")

# --- Візуалізація результатів (опціонально) ---
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')
plt.title('Фактичні vs Прогнозовані Рейтинги')
plt.xlabel('Фактичні Рейтинги')
plt.ylabel('Прогнозовані Рейтинги')
plt.show()
