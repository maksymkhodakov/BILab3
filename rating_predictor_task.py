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


# Функція для отримання введених даних від користувача
def get_user_input():
    print("Введіть дані для прогнозування рейтингу книги:")
    publish_year = int(input("Введіть рік публікації: "))
    language = input("Введіть мову (наприклад, 'eng'): ")
    title = input("Введіть назву книги: ")
    # Отримуємо ознаки з назви
    letter_features = extract_letter_features(title)
    return publish_year, language, letter_features


# --- Завантаження та підготовка даних ---

# Завантаження даних з CSV; роздільник ';'
df = pd.read_csv('CSV_BI_Lab1_data_source.csv', sep=';')

# Прибираємо зайві пробіли в назвах колонок, якщо потрібно
df.columns = [col.strip() for col in df.columns]

# Замінюємо кому на крапку в колонці 'Rating' та переводимо у тип float
df['Rating'] = df['Rating'].replace({',': '.'}, regex=True).astype(float)

# Видаляємо записи з пропущеним рейтингом
df = df.dropna(subset=['Rating'])

# Використовуємо наступні ознаки для прогнозу:
# - PublishYear (рік публікації)
# - Language (мова)
# - Літери, присутні в назві (Name)
# Для цього спочатку створимо ознаки з назви книги.
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

# Видаляємо можливі NaN (хоча зазвичай їх не має після імпутації)
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

# --- Отримання введених даних від користувача та прогноз ---
publish_year, language, user_letter_features = get_user_input()

# Формуємо DataFrame для введених даних
user_data = pd.DataFrame({
    'PublishYear': [publish_year]
})

# One-hot кодування введеної мови (маємо використовувати вже навчений encoder)
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

# Прогноз
prediction = model.predict(user_input_final)
print(f"\nПрогнозований рейтинг книги: {prediction[0]:.2f}")

# --- Візуалізація результатів ---
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')
plt.title('Фактичні vs Прогнозовані Рейтинги')
plt.xlabel('Фактичні Рейтинги')
plt.ylabel('Прогнозовані Рейтинги')
plt.show()
