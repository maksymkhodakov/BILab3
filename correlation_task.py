import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from scipy.stats import chi2_contingency

# Завантаження даних
df = pd.read_csv('CSV_BI_Lab1_data_source.csv', sep=';')

# Заповнення пропусків для категоріальних змінних 'Language', 'Authors', 'PublisherNaming'
df['Language'] = df['Language'].fillna(df['Language'].mode()[0])  # Заповнюємо пропуски найбільш частим значенням
df['Authors'] = df['Authors'].fillna(df['Authors'].mode()[0])  # Аналогічно для 'Authors'
df['PublisherNaming'] = df['PublisherNaming'].fillna(df['PublisherNaming'].mode()[0])  # Аналогічно для 'PublisherNaming'

# Створюємо категоріальну змінну для 'CountsOfReview' (для аналізу залежностей)
df['CountsOfReviewCategory'] = pd.cut(df['CountsOfReview'], bins=[0, 100, 500, 1000, 5000, 10000, 20000],
                                      labels=["0-100", "101-500", "501-1000", "1001-5000", "5001-10000", "10001-20000"])

# Побудова графіків для категоріальних змінних
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Language', palette='Set2')
plt.title('Розподіл по мовах')
plt.ylabel('Кількість книг')
plt.show()

# Побудова графіків для авторів
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Authors', palette='Set2', order=df['Authors'].value_counts().index[:10])  # Топ 10 авторів
plt.title('Топ 10 авторів')
plt.ylabel('Кількість книг')
plt.xticks(rotation=90)
plt.show()

# Побудова діаграми для категоріальної змінної 'CountsOfReviewCategory'
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='CountsOfReviewCategory', palette='Set3')
plt.title('Розподіл по категоріях кількості оглядів')
plt.ylabel('Кількість книг')
plt.show()

# Аналіз залежності між мовою та кількістю оглядів за допомогою хі-квадрат тесту
crosstab = pd.crosstab(df['Language'], df['CountsOfReviewCategory'])
chi2, p_val, _, _ = chi2_contingency(crosstab)

# Виведення результатів хі-квадрат тесту
print(f"\nChi-squared test між мовою та категорією кількості оглядів:")
print(f"Chi-squared: {chi2}")
print(f"P-value: {p_val}")

if p_val < 0.05:
    print("Є статистична залежність між мовою та категорією кількості оглядів (р < 0.05).")
else:
    print("Немає статистичної залежності між мовою та категорією кількості оглядів (р >= 0.05).")

# Побудова графіків для числових змінних
plt.figure(figsize=(10, 6))
sns.histplot(df['CountsOfReview'], kde=True, color='blue', bins=30)
plt.title('Розподіл кількості оглядів')
plt.xlabel('Кількість оглядів')
plt.ylabel('Частота')
plt.show()

# Візуалізація залежності між кількістю оглядів та рейтингом
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='CountsOfReview', y='Rating', color='green')
plt.title('Залежність між кількістю оглядів та рейтингом')
plt.xlabel('Кількість оглядів')
plt.ylabel('Рейтинг')
plt.show()
