import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, cohen_kappa_score, log_loss

# ==============================
# 1. Завантаження та попередня обробка даних
# ==============================
# Завантаження даних із CSV-файлу (переконайтеся, що файл містить: Name, CountsOfReview, PublishMonth, PublishYear, Rating)
df = pd.read_csv('CSV_BI_Lab1_data_source.csv', sep=';')

# Видаляємо записи з пропущеними критичними ознаками
df = df.dropna(subset=['Name', 'CountsOfReview', 'PublishMonth', 'PublishYear', 'Rating'])

# Перетворюємо PublishYear та PublishMonth у цілі числа
df['PublishYear'] = df['PublishYear'].astype(int)
df['PublishMonth'] = df['PublishMonth'].astype(int)

# Обробка рейтингу: замінюємо кому на крапку та перетворюємо у float
df['Rating'] = df['Rating'].replace({',': '.'}, regex=True).astype(float)

# ==============================
# 2. Формування набору ознак та цільової змінної
# ==============================
# Цільова змінна: PublishMonth (класи від 1 до 12)
y = df['PublishMonth']

# Ознаки для класифікації: CountsOfReview, Rating, PublishYear
X = df[['CountsOfReview', 'Rating', 'PublishYear']].copy()

# Заповнюємо можливі пропуски (якщо є)
imputer = SimpleImputer(strategy='mean')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)

# Масштабування ознак
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==============================
# 3. Розподіл даних на тренувальну та тестову вибірки (з поверненням індексів)
# ==============================
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X_scaled, y, df.index, test_size=0.2, random_state=42, stratify=y
)

# ==============================
# 4. Навчання класифікаційної моделі
# ==============================
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Прогнозування на тестовій вибірці
y_pred = clf.predict(X_test)
y_pred_proba = clf.predict_proba(X_test)  # для log loss

# Оцінка якості моделі
accuracy = accuracy_score(y_test, y_pred)
print("=== Результати класифікації ===")
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Додаткові метрики
macro_f1 = f1_score(y_test, y_pred, average='macro')
micro_f1 = f1_score(y_test, y_pred, average='micro')
weighted_f1 = f1_score(y_test, y_pred, average='weighted')

macro_precision = precision_score(y_test, y_pred, average='macro')
micro_precision = precision_score(y_test, y_pred, average='micro')
weighted_precision = precision_score(y_test, y_pred, average='weighted')

macro_recall = recall_score(y_test, y_pred, average='macro')
micro_recall = recall_score(y_test, y_pred, average='micro')
weighted_recall = recall_score(y_test, y_pred, average='weighted')

kappa = cohen_kappa_score(y_test, y_pred)
ll = log_loss(y_test, y_pred_proba)

print("\nДодаткові метрики:")
print(f"Macro F1-score: {macro_f1:.4f}")
print(f"Micro F1-score: {micro_f1:.4f}")
print(f"Weighted F1-score: {weighted_f1:.4f}")
print(f"Macro Precision: {macro_precision:.4f}")
print(f"Micro Precision: {micro_precision:.4f}")
print(f"Weighted Precision: {weighted_precision:.4f}")
print(f"Macro Recall: {macro_recall:.4f}")
print(f"Micro Recall: {micro_recall:.4f}")
print(f"Weighted Recall: {weighted_recall:.4f}")
print(f"Cohen's Kappa: {kappa:.4f}")
print(f"Log Loss: {ll:.4f}")

# ==============================
# 5. Запис результатів класифікації у файл
# ==============================
results_df = pd.DataFrame({
    'Name': df.loc[idx_test, 'Name'],
    'Actual_PublishMonth': y_test,
    'Predicted_PublishMonth': y_pred
})
results_df.to_csv('results_classification.csv', index=False, encoding='utf-8-sig')
print("\nРезультати класифікації з назвою роботи, фактичним та прогнозованим місяцем збережено у файл 'results_classification.csv'.")

# ==============================
# 6. Візуалізація результатів
# ==============================
# 6.1 Матриця плутанини
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=sorted(y.unique()), yticklabels=sorted(y.unique()))
plt.xlabel("Прогнозований місяць")
plt.ylabel("Фактичний місяць")
plt.title("Матриця плутанини для класифікації місяців публікації")
plt.show()

# 6.2 Візуалізація важливості ознак
importances = clf.feature_importances_
feature_names = X.columns
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(8, 6))
sns.barplot(x=importances[indices], hue=importances[indices], y=[feature_names[i] for i in indices], palette='viridis')
plt.title("Важливість ознак")
plt.xlabel("Важливість")
plt.ylabel("Ознака")
plt.show()

# 6.3 Інтерпретація результатів класифікації
print("\n=== Інтерпретація результатів класифікації ===")
print("Модель класифікує книги за місяцем публікації на основі наступних параметрів:")
print(" - CountsOfReview: кількість відгуків (популярність книги).")
print(" - Rating: рейтинг книги (якість сприйняття).")
print(" - PublishYear: рік публікації (тенденції та тренди ринку).")
print("\nВажливість ознак:")
for i, name in enumerate(feature_names[indices]):
    print(f"  {i+1}. {name}: важливість = {importances[indices][i]:.3f}")
print("\nМатриця плутанини демонструє, наскільки добре модель розрізняє 12 класів (місяців).")
