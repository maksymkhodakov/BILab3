import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score, davies_bouldin_score
import numpy as np

# Завантаження даних
df = pd.read_csv('CSV_BI_Lab1_data_source.csv', sep=';')

# Видаляємо коми в колонці 'Rating' і перетворюємо її на тип float
df['Rating'] = df['Rating'].replace({',': '.'}, regex=True).astype(float)

# Обробка відсутніх значень
df = df.dropna(subset=['Rating'])

# Заповнення відсутніх значень в інших ознаках середнім значенням
imputer = SimpleImputer(strategy='mean')
df[['RatingDist1', 'RatingDist2', 'RatingDist3', 'RatingDist4', 'RatingDist5', 'CountsOfReview']] = imputer.fit_transform(df[['RatingDist1', 'RatingDist2', 'RatingDist3', 'RatingDist4', 'RatingDist5', 'CountsOfReview']])

# Вибір ознак для кластеризації
X = df[['RatingDist1', 'RatingDist2', 'RatingDist3', 'RatingDist4', 'RatingDist5']]

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
    print(cluster_data[['RatingDist1', 'RatingDist2', 'RatingDist3', 'RatingDist4', 'RatingDist5']].describe())

# Виведення розподілу кожного кластера по ознаках
print("\nРозподіл по кожному кластеру для ознак RatingDist1, RatingDist2:")
print(df.groupby('KMeans_Cluster')[['RatingDist1', 'RatingDist2']].mean())

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

# Візуалізація результатів кластеризації KMeans (1)
plt.figure(figsize=(8, 6))
plt.scatter(df['RatingDist1'], df['RatingDist2'], c=df['KMeans_Cluster'], cmap='viridis', alpha=0.6)
plt.title('KMeans Clustering (RatingDist1 vs RatingDist2)')
plt.xlabel('RatingDist1')
plt.ylabel('RatingDist2')
plt.colorbar(label='Cluster')
plt.show()

# Візуалізація результатів кластеризації KMeans (2)
plt.figure(figsize=(8, 6))
plt.scatter(df['RatingDist3'], df['RatingDist4'], c=df['KMeans_Cluster'], cmap='viridis', alpha=0.6)
plt.title('KMeans Clustering (RatingDist3 vs RatingDist4)')
plt.xlabel('RatingDist3')
plt.ylabel('RatingDist4')
plt.colorbar(label='Cluster')
plt.show()

# Візуалізація результатів кластеризації KMeans з центрами кластерів
plt.figure(figsize=(8, 6))
plt.scatter(df['RatingDist1'], df['RatingDist2'], c=df['KMeans_Cluster'], cmap='viridis', alpha=0.6)
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=200, label='Cluster Centers')
plt.title('KMeans Clustering with Centers (RatingDist1 vs RatingDist2)')
plt.xlabel('RatingDist1')
plt.ylabel('RatingDist2')
plt.colorbar(label='Cluster')
plt.legend()
plt.show()

# Додатково: відображення перших кількох рядків даних з класифікацією
print("\nПерші кілька рядків з класифікацією:")
print(df[['RatingDist1', 'RatingDist2', 'KMeans_Cluster']].head())
