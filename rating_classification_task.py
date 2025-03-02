import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# Вмикаємо інтерактивний режим для графіків
plt.ion()

# --- Завантаження та попередня обробка даних ---
usecols = ['Name', 'Rating', 'PublishYear', 'PublisherNaming']
# Для тестування завантажуємо лише 10000 рядків
df = pd.read_csv('CSV_BI_Lab1_data_source.csv', sep=';', usecols=usecols, nrows=10000)

df['Rating'] = df['Rating'].replace({',': '.'}, regex=True).astype(float)
df = df.dropna(subset=['Rating'])
df['RatingClass'] = np.where(df['Rating'] >= 4.5, 'High', 'Low')

# --- Обробка PublisherNaming ---
df['PublisherNaming'] = df['PublisherNaming'].fillna("Unknown")
min_freq = 10
pub_counts = df['PublisherNaming'].value_counts()
df['PublisherNaming'] = df['PublisherNaming'].apply(lambda x: x if pub_counts[x] >= min_freq else 'Other')

# --- Формування ознак ---
imputer = SimpleImputer(strategy='mean')
df[['PublishYear']] = imputer.fit_transform(df[['PublishYear']])

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
publisher_encoded = encoder.fit_transform(df[['PublisherNaming']])
publisher_encoded_df = pd.DataFrame(publisher_encoded,
                                    columns=encoder.get_feature_names_out(['PublisherNaming']),
                                    index=df.index)

X = pd.concat([df[['PublishYear']], publisher_encoded_df], axis=1)
y = df['RatingClass']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Побудова моделей класифікації ---
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
y_pred_log = log_reg.predict(X_test)
accuracy_log = accuracy_score(y_test, y_pred_log)

print("=== Logistic Regression Results ===")
print(f"Accuracy: {accuracy_log:.2f}\n")
cm_log = confusion_matrix(y_test, y_pred_log, labels=['High', 'Low'])
print("Confusion Matrix (Logistic Regression):")
print(cm_log)
print("\nClassification Report (Logistic Regression):")
print(classification_report(y_test, y_pred_log, target_names=['High', 'Low']))

tree_clf = DecisionTreeClassifier(random_state=42)
tree_clf.fit(X_train, y_train)
y_pred_tree = tree_clf.predict(X_test)
accuracy_tree = accuracy_score(y_test, y_pred_tree)

print("\n=== Decision Tree Classifier Results ===")
print(f"Accuracy: {accuracy_tree:.2f}\n")
cm_tree = confusion_matrix(y_test, y_pred_tree, labels=['High', 'Low'])
print("Confusion Matrix (Decision Tree):")
print(cm_tree)
print("\nClassification Report (Decision Tree):")
print(classification_report(y_test, y_pred_tree, target_names=['High', 'Low']))


def plot_confusion_matrix(cm, title):
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['High', 'Low'], yticklabels=['High', 'Low'])
    plt.title(title)
    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")
    plt.show()


plot_confusion_matrix(cm_log, "Confusion Matrix - Logistic Regression")
plot_confusion_matrix(cm_tree, "Confusion Matrix - Decision Tree")

plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='RatingClass', palette='viridis')
plt.title("Кількість книг за класами RatingClass")
plt.xlabel("Клас рейтингу")
plt.ylabel("Кількість книг")
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='PublishYear', hue='RatingClass', palette='viridis')
plt.title("Розподіл класів за роками публікації")
plt.xlabel("Рік публікації")
plt.ylabel("Кількість книг")
plt.xticks(rotation=45)
plt.legend(title='Rating Class')
plt.show()

print("\nПерші 10 рядків даних з колонкою 'RatingClass':")
print(df[['Name', 'PublishYear', 'PublisherNaming', 'Rating', 'RatingClass']].head(10))

classification_results = df[['Name', 'RatingClass']]
classification_results.to_csv('classification_results.csv', index=False, encoding='utf-8-sig')
print("\nРезультати класифікації збережено у файл 'classification_results.csv'")
