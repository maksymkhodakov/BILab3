import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.impute import SimpleImputer

# Завантаження даних
df = pd.read_csv('CSV_BI_Lab1_data_source.csv', sep=';')

# Обробка колонки 'Rating': замінюємо коми на крапки та переводимо у тип float
df['Rating'] = df['Rating'].replace({',': '.'}, regex=True).astype(float)

# Видаляємо записи з пропущеним рейтингом
df = df.dropna(subset=['Rating'])

# Створення цільової змінної: якщо Rating >= 4.5, то "High", інакше "Low"
df['RatingClass'] = np.where(df['Rating'] >= 4.5, 'High', 'Low')

# Вибір ознак для моделі: розподіл оцінок та загальна кількість відгуків
features = ['RatingDist1', 'RatingDist2', 'RatingDist3', 'RatingDist4', 'RatingDist5', 'CountsOfReview']

# Заповнення пропущених значень середнім значенням для ознак
imputer = SimpleImputer(strategy='mean')
df[features] = imputer.fit_transform(df[features])

# Формування X (ознаки) та y (цільова змінна)
X = df[features]
y = df['RatingClass']

# Розподіл даних на тренувальну та тестову вибірки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Модель 1: Logistic Regression ---
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
y_pred_log = log_reg.predict(X_test)
accuracy_log = accuracy_score(y_test, y_pred_log)

# Детальний звіт для Logistic Regression
print("=== Logistic Regression Results ===")
print(f"Accuracy: {accuracy_log:.2f}\n")
cm_log = confusion_matrix(y_test, y_pred_log, labels=['High','Low'])
print("Confusion Matrix (Logistic Regression):")
print(cm_log)
print("\nClassification Report (Logistic Regression):")
print(classification_report(y_test, y_pred_log, target_names=['High', 'Low']))

# --- Модель 2: Decision Tree Classifier ---
tree_clf = DecisionTreeClassifier(random_state=42)
tree_clf.fit(X_train, y_train)
y_pred_tree = tree_clf.predict(X_test)
accuracy_tree = accuracy_score(y_test, y_pred_tree)

# Детальний звіт для Decision Tree
print("\n=== Decision Tree Classifier Results ===")
print(f"Accuracy: {accuracy_tree:.2f}\n")
cm_tree = confusion_matrix(y_test, y_pred_tree, labels=['High','Low'])
print("Confusion Matrix (Decision Tree):")
print(cm_tree)
print("\nClassification Report (Decision Tree):")
print(classification_report(y_test, y_pred_tree, target_names=['High', 'Low']))

# --- Візуалізація результатів класифікації ---

# Функція для побудови heatmap матриці плутанини
def plot_confusion_matrix(cm, title):
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['High','Low'], yticklabels=['High','Low'])
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

plot_confusion_matrix(cm_log, "Confusion Matrix - Logistic Regression")
plot_confusion_matrix(cm_tree, "Confusion Matrix - Decision Tree")

# Візуалізація розподілу кількості книг за класами
plt.figure(figsize=(6,4))
sns.countplot(data=df, x='RatingClass', palette='viridis')
plt.title("Кількість книг за класами RatingClass")
plt.xlabel("Rating Class")
plt.ylabel("Кількість книг")
plt.show()

# Візуалізація середніх значень ознак для кожного класу
class_summary = df.groupby('RatingClass')[features].mean().reset_index()
class_summary = pd.melt(class_summary, id_vars='RatingClass', var_name='Feature', value_name='Mean Value')

plt.figure(figsize=(10,6))
sns.barplot(data=class_summary, x='Feature', y='Mean Value', hue='RatingClass', palette='viridis')
plt.title("Середні значення ознак для класів High та Low")
plt.xticks(rotation=45)
plt.ylabel("Середнє значення")
plt.xlabel("Ознака")
plt.show()

# Виведення перших 10 рядків даних з класифікацією
print("\nПерші 10 рядків даних з колонкою 'RatingClass':")
print(df[['Name', 'RatingDist1', 'RatingDist2', 'RatingDist3', 'RatingDist4', 'RatingDist5', 'CountsOfReview', 'Rating', 'RatingClass']].head(10))

# --- Збереження результатів класифікації в CSV ---
# Зберігаємо лише стовпці 'Name' та 'RatingClass'
classification_results = df[['Name', 'RatingClass']]
classification_results.to_csv('classification_results.csv', index=False, encoding='utf-8-sig')
print("\nРезультати класифікації збережено у файл 'classification_results.csv'")
