import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score, davies_bouldin_score
import numpy as np
from sklearn.decomposition import PCA

# Завантаження даних
df = pd.read_csv('CSV_BI_Lab1_data_source.csv', sep=';')

# Обробка відсутніх значень
df = df.dropna(subset=['Language', 'CountsOfReview'])  # Видаляємо рядки з пропущеними значеннями в Language та CountsOfReview

# Перетворення категоріальної змінної 'Language' у числову за допомогою OneHotEncoder
encoder = OneHotEncoder(sparse_output=False)

# Перетворення Language в числові ознаки
language_encoded = encoder.fit_transform(df[['Language']])

# Створення DataFrame для закодованих колонок
language_encoded_df = pd.DataFrame(language_encoded, columns=encoder.get_feature_names_out(['Language']))

# Додавання закодованих колонок до основного DataFrame
df = pd.concat([df, language_encoded_df], axis=1)

# Заповнення пропущених значень для числових колонок, включаючи CountsOfReview
imputer = SimpleImputer(strategy='mean')
df[['CountsOfReview'] + language_encoded_df.columns.tolist()] = imputer.fit_transform(df[['CountsOfReview'] + language_encoded_df.columns.tolist()])

# Вибір ознак для кластеризації
X = df[language_encoded_df.columns.tolist() + ['CountsOfReview']]

# Нормалізація ознак
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Кластеризація за допомогою KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)
df['KMeans_Cluster'] = kmeans.labels_

# Виведення інформації про кількість елементів в кожному кластері
print("Кількість елементів в кожному кластері:")
print(df['KMeans_Cluster'].value_counts())

# Виведення координат центрів кластерів
print("\nЦентри кластерів (KMeans):")
print(kmeans.cluster_centers_)

# Статистика по кожному кластеру для кожної ознаки
print("\nСтатистика по кожному кластеру:")
for cluster in range(3):  # 3 кластери, як визначено в KMeans
    cluster_data = df[df['KMeans_Cluster'] == cluster]
    print(f"\nКластер {cluster}:")
    print(cluster_data[language_encoded_df.columns.tolist() + ['CountsOfReview']].describe())

# Виведення розподілу кожного кластера по ознаках
print("\nРозподіл по кожному кластеру для ознак Language та CountsOfReview:")
print(df.groupby('KMeans_Cluster')[language_encoded_df.columns.tolist() + ['CountsOfReview']].mean())

# Додаткові метрики:
# Використання підвибірки (наприклад, 10% даних)
sample_size = int(0.1 * X_scaled.shape[0])  # 10% від усіх даних
sample_indices = np.random.choice(X_scaled.shape[0], sample_size, replace=False)
X_sampled = X_scaled[sample_indices]

# 1. Silhouette Score
silhouette = silhouette_score(X_sampled, kmeans.labels_[sample_indices])
print(f"\nSilhouette Score: {silhouette:.4f}")

# 2. Davies-Bouldin Index
davies_bouldin = davies_bouldin_score(X_sampled, kmeans.labels_[sample_indices])
print(f"\nDavies-Bouldin Index: {davies_bouldin:.4f}")

# 3. Inertia (WSS)
inertia = kmeans.inertia_
print(f"\nInertia (WSS): {inertia:.4f}")

# Візуалізація результатів кластеризації KMeans
# Використовуємо PCA для зменшення вимірності до 2D для зручнішої візуалізації
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Візуалізація кластерів з центрами
plt.figure(figsize=(10, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df['KMeans_Cluster'], cmap='viridis', alpha=0.6)
centers = pca.transform(kmeans.cluster_centers_)
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=200, label='Cluster Centers')

# Покращення графіка:
plt.title('KMeans Clustering with Centers (PCA Projection)', fontsize=16)
plt.xlabel('PCA Component 1', fontsize=12)
plt.ylabel('PCA Component 2', fontsize=12)
plt.colorbar(label='Cluster')
plt.legend()
plt.grid(True)
plt.show()

# Візуалізація для розподілу "Language" по кластерах
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Language', hue='KMeans_Cluster', palette='Set2')
plt.title('Розподіл по мовах в кожному кластері')
plt.ylabel('Кількість книг')
plt.xlabel('Мова')
plt.xticks(rotation=45)
plt.show()

# Візуалізація для розподілу "CountsOfReview" по кластерах
plt.figure(figsize=(10, 6))
sns.boxplot(x='KMeans_Cluster', y='CountsOfReview', data=df, palette='Set3')
plt.title('Розподіл кількості оглядів по кластерах')
plt.xlabel('Кластер')
plt.ylabel('Кількість оглядів')
plt.show()

# Додатково: відображення перших кількох рядків даних з класифікацією
print("\nПерші кілька рядків з класифікацією:")
print(df[['Language', 'CountsOfReview', 'KMeans_Cluster']].head())
