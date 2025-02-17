import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency, spearmanr
from sklearn.impute import SimpleImputer
from sklearn.metrics import mutual_info_score

# Завантаження даних
df = pd.read_csv('CSV_BI_Lab1_data_source.csv', sep=';')

# Заповнення пропусків для категоріальних змінних 'Language', 'Authors', 'PublisherNaming'
df['Language'] = df['Language'].fillna(df['Language'].mode()[0])
df['Authors'] = df['Authors'].fillna(df['Authors'].mode()[0])
df['PublisherNaming'] = df['PublisherNaming'].fillna(df['PublisherNaming'].mode()[0])

# Перетворення значень у колонці 'Rating': заміна ком на крапки та перетворення на float
df['Rating'] = df['Rating'].replace({',': '.'}, regex=True).astype(float)

# Створення категоріальної змінної для 'CountsOfReview'
df['CountsOfReviewCategory'] = pd.cut(df['CountsOfReview'], bins=[0, 100, 500, 1000, 5000, 10000, 20000],
                                      labels=["0-100", "101-500", "501-1000", "1001-5000", "5001-10000", "10001-20000"])

# Заповнення пропущених значень для числових колонок, якщо є
imputer = SimpleImputer(strategy='mean')
df[['CountsOfReview', 'Rating']] = imputer.fit_transform(df[['CountsOfReview', 'Rating']])


# --- Аналіз залежностей за допомогою статистичних тестів ---

def chi_square_test(column1, column2):
    crosstab = pd.crosstab(column1, column2)
    chi2, p_val, _, _ = chi2_contingency(crosstab)
    print(f"\nChi-squared test між {column1.name} та {column2.name}:")
    print(f"Chi-squared: {chi2:.2f}")
    print(f"P-value: {p_val:.4f}")
    if p_val < 0.05:
        print("Є статистична залежність між цими змінними (р < 0.05).")
    else:
        print("Немає статистичної залежності між цими змінними (р >= 0.05).")


# Тести залежностей
chi_square_test(df['Language'], df['CountsOfReviewCategory'])
chi_square_test(df['Authors'], df['CountsOfReviewCategory'])
chi_square_test(df['PublisherNaming'], df['CountsOfReviewCategory'])

# Pearson кореляція (для числових змінних)
pearson_corr = df[['CountsOfReview', 'Rating']].corr(method='pearson')
print("\nPearson Correlation Matrix:")
print(pearson_corr)

# Spearman кореляція (для монотонних залежностей)
spearman_corr, _ = spearmanr(df['CountsOfReview'], df['Rating'])
print(f"\nSpearman Correlation between 'CountsOfReview' and 'Rating': {spearman_corr:.4f}")

# Mutual Information (для нелінійних залежностей)
df['CountsOfReviewCategory'] = df['CountsOfReviewCategory'].fillna(df['CountsOfReviewCategory'].mode()[0])
mi = mutual_info_score(df['CountsOfReviewCategory'], df['Rating'])
print(f"\nMutual Information between 'CountsOfReviewCategory' and 'Rating': {mi:.4f}")

# --- Розширена візуалізація ---

# 1. Візуалізація розподілу категоріальних змінних

# Розподіл по мовах
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Language', palette='Set2', order=df['Language'].value_counts().index)
plt.title('Розподіл книг по мовах')
plt.xlabel('Мова')
plt.ylabel('Кількість книг')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Топ 10 авторів
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Authors', palette='Set2', order=df['Authors'].value_counts().index[:10])
plt.title('Топ 10 авторів')
plt.xlabel('Автор')
plt.ylabel('Кількість книг')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Розподіл по категоріям кількості оглядів
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='CountsOfReviewCategory', palette='Set3',
              order=df['CountsOfReviewCategory'].value_counts().index)
plt.title('Розподіл книг по категоріях кількості оглядів')
plt.xlabel('Категорія кількості оглядів')
plt.ylabel('Кількість книг')
plt.tight_layout()
plt.show()

# 2. Візуалізація числових змінних

# Гістограма кількості оглядів з KDE
plt.figure(figsize=(10, 6))
sns.histplot(df['CountsOfReview'], kde=True, color='blue', bins=30)
plt.title('Розподіл кількості оглядів')
plt.xlabel('Кількість оглядів')
plt.ylabel('Частота')
plt.tight_layout()
plt.show()

# Scatter plot: залежність між кількістю оглядів та рейтингом
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='CountsOfReview', y='Rating', color='green', alpha=0.7)
plt.title('Залежність між кількістю оглядів та рейтингом')
plt.xlabel('Кількість оглядів')
plt.ylabel('Рейтинг')
plt.tight_layout()
plt.show()

# Boxplot: розподіл рейтингу по категоріях кількості оглядів
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='CountsOfReviewCategory', y='Rating', palette='Set1')
plt.title('Розподіл рейтингу по категоріях кількості оглядів')
plt.xlabel('Категорія кількості оглядів')
plt.ylabel('Рейтинг')
plt.tight_layout()
plt.show()

# 3. Додаткова візуалізація залежностей через теплову карту кореляцій для числових змінних
plt.figure(figsize=(8, 6))
corr_matrix = df[['CountsOfReview', 'Rating']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Кореляційна матриця для CountsOfReview та Rating')
plt.tight_layout()
plt.show()

# --- Business Value та детальна інтерпретація ---

print("\n=== Business Value ===")
print("1. Аналіз категоріальних змінних (Language, Authors, PublisherNaming, CountsOfReviewCategory):")
print("- Дозволяє видавцям визначити, які сегменти ринку (мовні, авторські або за видавництвом) є найбільш активними.")
print("   - Ця інформація може бути використана для таргетованої реклами та оптимізації маркетингових кампаній.")
print("2. Залежність між кількістю оглядів та рейтингом:")
print("- Кореляційний аналіз та візуалізації допомагають зрозуміти, як кількість відгуків впливає на загальний "
      "рейтинг книги.")
print("   - Це може вплинути на стратегію підвищення якості продукту та роботу з клієнтами.")
print("3. Візуалізація розподілу даних:")
print("- Графіки (histogram, scatter plot, boxplot, heatmap) забезпечують наочність даних і дозволяють виявити "
      "закономірності,")
print("     які важливі для прийняття бізнес-рішень, таких як планування випусків та розподіл рекламного бюджету.")
print("4. Використання статистичних тестів (Chi-squared, Spearman, Mutual Information):")
print("- Ці показники допомагають об'єктивно оцінити залежності між змінними та визначити, чи існують статистично "
      "значущі зв'язки.")
print("   - Результати тестів можна використовувати для подальшої сегментації ринку та оптимізації бізнес-процесів.")
print("5. Загалом, цей аналіз дозволяє:")
print("   - Зрозуміти ринкові тренди та змінність вподобань клієнтів.")
print("   - Оптимізувати продуктовий портфель та спрямувати зусилля на найбільш перспективні сегменти.")
print("   - Приймати обґрунтовані рішення щодо майбутніх інвестицій та маркетингових стратегій.\n")
