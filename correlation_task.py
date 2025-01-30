import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from scipy.stats import chi2_contingency, pearsonr, spearmanr
from sklearn.metrics import mutual_info_score
import numpy as np

# Завантаження даних
df = pd.read_csv('CSV_BI_Lab1_data_source.csv', sep=';')

# Заповнення пропусків для категоріальних змінних 'Language', 'Authors', 'PublisherNaming'
df['Language'] = df['Language'].fillna(df['Language'].mode()[0])  # Заповнюємо пропуски найбільш частим значенням
df['Authors'] = df['Authors'].fillna(df['Authors'].mode()[0])  # Аналогічно для 'Authors'
df['PublisherNaming'] = df['PublisherNaming'].fillna(
    df['PublisherNaming'].mode()[0])  # Аналогічно для 'PublisherNaming'

# Перетворення значень у колонці 'Rating' з комами на крапки і перетворення на float
df['Rating'] = df['Rating'].replace({',': '.'}, regex=True).astype(float)

# Створюємо категоріальну змінну для 'CountsOfReview' (для аналізу залежностей)
df['CountsOfReviewCategory'] = pd.cut(df['CountsOfReview'], bins=[0, 100, 500, 1000, 5000, 10000, 20000],
                                      labels=["0-100", "101-500", "501-1000", "1001-5000", "5001-10000", "10001-20000"])


# Заповнення пропущених значень для числових колонок (якщо є пропущені значення)
imputer = SimpleImputer(strategy='mean')
df[['CountsOfReview', 'Rating']] = imputer.fit_transform(df[['CountsOfReview', 'Rating']])


# Функція для проведення хі-квадрат тесту
def chi_square_test(column1, column2):
    crosstab = pd.crosstab(column1, column2)
    chi2, p_val, _, _ = chi2_contingency(crosstab)
    print(f"\nChi-squared test між {column1.name} та {column2.name}:")
    print(f"Chi-squared: {chi2}")
    print(f"P-value: {p_val}")
    if p_val < 0.05:
        print("Є статистична залежність між цими змінними (р < 0.05).")
    else:
        print("Немає статистичної залежності між цими змінними (р >= 0.05).")


# Перевірка залежностей між кількома парами колонок
chi_square_test(df['Language'], df['CountsOfReviewCategory'])  # Залежність між мовою та категорією кількості оглядів
chi_square_test(df['Authors'], df['CountsOfReviewCategory'])  # Залежність між авторами та категорією кількості оглядів
chi_square_test(df['PublisherNaming'],
                df['CountsOfReviewCategory'])  # Залежність між видавцем та категорією кількості оглядів

# 1. Pearson Correlation between numerical variables
correlation_matrix = df[['CountsOfReview', 'Rating']].corr(method='pearson')
print(f"\nPearson Correlation Matrix:")
print(correlation_matrix)

# 2. Spearman Correlation (for monotonic relationships)
spearman_corr, _ = spearmanr(df['CountsOfReview'], df['Rating'])
print(f"\nSpearman Correlation between 'CountsOfReview' and 'Rating': {spearman_corr:.4f}")

# 3. Mutual Information (for non-linear relationships)
# Заповнюємо пропущені значення перед розрахунком
df['CountsOfReviewCategory'] = df['CountsOfReviewCategory'].fillna(
    df['CountsOfReviewCategory'].mode()[0])  # заповнення категоріальної змінної
mi = mutual_info_score(df['CountsOfReviewCategory'], df['Rating'])
print(f"\nMutual Information between 'CountsOfReviewCategory' and 'Rating': {mi:.4f}")

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
