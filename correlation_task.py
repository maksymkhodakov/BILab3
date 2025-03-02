import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, spearmanr
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import LabelEncoder

# ------------------------------
# 1. Завантаження та попередня обробка даних
# ------------------------------
# Завантаження даних із CSV-файлу
df = pd.read_csv('CSV_BI_Lab1_data_source.csv', sep=';')

# Обробка рейтингу: заміна ком на крапки та перетворення в float
df['Rating'] = df['Rating'].replace({',': '.'}, regex=True).astype(float)

# Перетворення важливих колонок у числовий формат
numeric_cols = ['CountsOfReview', 'Rating', 'PublishYear', 'PublishMonth']
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

# Видаляємо рядки з пропущеними значеннями для цих колонок
df = df.dropna(subset=numeric_cols)

# ------------------------------
# 2. Додаткові характеристики
# ------------------------------
# 2.1 Довжина назви книги (TitleLength)
df['TitleLength'] = df['Name'].fillna("").apply(len)

# 2.2 Для видавництва (PublisherNaming) переконаємося, що немає пропусків
df['PublisherNaming'] = df['PublisherNaming'].fillna(df['PublisherNaming'].mode()[0])

# ------------------------------
# 3. Обчислення кореляційних матриць для числових змінних
# ------------------------------
# Розширений набір числових характеристик
num_features = ['CountsOfReview', 'Rating', 'PublishYear', 'PublishMonth', 'TitleLength']

# 3.1 Pearson кореляція
pearson_corr = df[num_features].corr(method='pearson')
print("Pearson Correlation Matrix (Extended):")
print(pearson_corr)

# 3.2 Spearman кореляція
spearman_corr = df[num_features].corr(method='spearman')
print("\nSpearman Correlation Matrix (Extended):")
print(spearman_corr)

# 3.3 Mutual Information (для нелінійних залежностей)
mi_matrix = pd.DataFrame(index=num_features, columns=num_features)
for col1 in num_features:
    for col2 in num_features:
        mi = mutual_info_regression(df[[col1]], df[col2], random_state=42)
        mi_matrix.loc[col1, col2] = mi[0]
mi_matrix = mi_matrix.astype(float)
print("\nMutual Information Matrix (Extended):")
print(mi_matrix)

# ------------------------------
# 4. Аналіз PublisherNaming (категорійна змінна)
# ------------------------------
# Використаємо χ²-тест для перевірки залежності між PublisherNaming та, наприклад, категоріями кількості відгуків.
df['CountsOfReviewCategory'] = pd.cut(df['CountsOfReview'],
                                      bins=[0, 100, 500, 1000, 5000, 10000, 20000],
                                      labels=["0-100", "101-500", "501-1000", "1001-5000", "5001-10000", "10001-20000"])
ct = pd.crosstab(df['PublisherNaming'], df['CountsOfReviewCategory'])
chi2, p_val, dof, expected = chi2_contingency(ct)
print("\nChi-squared test between PublisherNaming and CountsOfReviewCategory:")
print(f"Chi-squared: {chi2:.2f}, P-value: {p_val:.4f}")

# Також можна обчислити взаємну інформацію між PublisherNaming та іншими числовими змінними,
# спершу проведемо Label Encoding для PublisherNaming
le = LabelEncoder()
df['PublisherEncoded'] = le.fit_transform(df['PublisherNaming'])
mi_publisher = {}
for feature in ['CountsOfReview', 'Rating', 'PublishYear', 'PublishMonth', 'TitleLength']:
    mi = mutual_info_regression(df[[feature]], df['PublisherEncoded'], random_state=42)
    mi_publisher[feature] = mi[0]
print("\nMutual Information between PublisherNaming and numeric features:")
print(mi_publisher)

# ------------------------------
# 5. Візуалізація кореляційних матриць та парних залежностей
# ------------------------------
plt.figure(figsize=(10, 8))
sns.heatmap(pearson_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Pearson Correlation Matrix (Extended)')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(spearman_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Spearman Correlation Matrix (Extended)')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(mi_matrix, annot=True, cmap='viridis')
plt.title('Mutual Information Matrix (Extended)')
plt.tight_layout()
plt.show()
