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

# Видаляємо записи з пропущеними значеннями для важливих колонок
df = df.dropna(subset=['Language', 'CountsOfReview', 'PublishYear'])

# Переконуємося, що PublishYear має числовий тип
df['PublishYear'] = df['PublishYear'].astype(int)

# Перетворення категоріальної змінної 'Language' у числову за допомогою OneHotEncoder
encoder = OneHotEncoder(sparse_output=False)
language_encoded = encoder.fit_transform(df[['Language']])
language_encoded_df = pd.DataFrame(language_encoded, columns=encoder.get_feature_names_out(['Language']))

# Додавання закодованих колонок до основного DataFrame
df = pd.concat([df, language_encoded_df], axis=1)

# Заповнення пропущених значень для числових колонок (CountsOfReview та закодовані колонки)
imputer = SimpleImputer(strategy='mean')
cols_to_impute = ['CountsOfReview'] + language_encoded_df.columns.tolist()
df[cols_to_impute] = imputer.fit_transform(df[cols_to_impute])

# Вибір ознак для кластеризації: закодована мова + кількість відгуків
X = df[language_encoded_df.columns.tolist() + ['CountsOfReview']]

# Нормалізація ознак
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Кластеризація за допомогою KMeans (3 кластери)
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
for cluster in range(3):
    cluster_data = df[df['KMeans_Cluster'] == cluster]
    print(f"\nКластер {cluster}:")
    print(cluster_data[language_encoded_df.columns.tolist() + ['CountsOfReview']].describe())

# Розподіл по кожному кластеру для ознак Language та CountsOfReview
print("\nРозподіл по кожному кластеру для ознак Language та CountsOfReview:")
print(df.groupby('KMeans_Cluster')[language_encoded_df.columns.tolist() + ['CountsOfReview']].mean())

# Додаткові метрики:
sample_size = int(0.1 * X_scaled.shape[0])  # 10% від усіх даних
sample_indices = np.random.choice(X_scaled.shape[0], sample_size, replace=False)
X_sampled = X_scaled[sample_indices]
silhouette = silhouette_score(X_sampled, kmeans.labels_[sample_indices])
davies_bouldin = davies_bouldin_score(X_sampled, kmeans.labels_[sample_indices])
inertia = kmeans.inertia_
print(f"\nSilhouette Score: {silhouette:.4f}")
print(f"\nDavies-Bouldin Index: {davies_bouldin:.4f}")
print(f"\nInertia (WSS): {inertia:.4f}")

# Візуалізація результатів кластеризації за допомогою PCA (зменшення вимірності до 2D)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
plt.figure(figsize=(10, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df['KMeans_Cluster'], cmap='viridis', alpha=0.6)
centers = pca.transform(kmeans.cluster_centers_)
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=200, label='Cluster Centers')
plt.title('KMeans Clustering with Centers (PCA Projection)', fontsize=16)
plt.xlabel('PCA Component 1', fontsize=12)
plt.ylabel('PCA Component 2', fontsize=12)
plt.colorbar(label='Cluster')
plt.legend()
plt.grid(True)
plt.show()

# Візуалізація розподілу мов по кластерах
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Language', hue='KMeans_Cluster', palette='Set2')
plt.title('Розподіл по мовах в кожному кластері')
plt.ylabel('Кількість книг')
plt.xlabel('Мова')
plt.xticks(rotation=45)
plt.show()

# Візуалізація розподілу "CountsOfReview" по кластерах
plt.figure(figsize=(10, 6))
sns.boxplot(x='KMeans_Cluster', y='CountsOfReview', data=df, palette='Set3')
plt.title('Розподіл кількості оглядів по кластерах')
plt.xlabel('Кластер')
plt.ylabel('Кількість оглядів')
plt.show()

# Додаткова сегментація: Групування даних за роками публікації
year_cluster = df.groupby(['PublishYear', 'KMeans_Cluster']).size().reset_index(name='Count')
print("\nРозподіл кластерів по роках публікації:")
print(year_cluster)

# Візуалізація розподілу кластерів за роками (barplot)
plt.figure(figsize=(12, 6))
year_pivot = year_cluster.pivot(index='PublishYear', columns='KMeans_Cluster', values='Count').fillna(0)
year_pivot.plot(kind='bar', stacked=True, colormap='viridis', figsize=(12, 6))
plt.title('Розподіл кластерів за роками публікації')
plt.xlabel('Рік публікації')
plt.ylabel('Кількість книг')
plt.legend(title='Кластер')
plt.grid(axis='y')
plt.show()

# Візуалізація середніх значень CountsOfReview за роками для кожного кластера
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x='PublishYear', y='CountsOfReview', hue='KMeans_Cluster', marker='o', palette='viridis')
plt.title('Середня кількість відгуків по кластерах за роками публікації')
plt.xlabel('Рік публікації')
plt.ylabel('CountsOfReview')
plt.grid(True)
plt.show()

# Додатково: відображення перших кількох рядків даних з кластеризацією
print("\nПерші кілька рядків з класифікацією:")
print(df[['Name', 'PublishYear', 'Language', 'CountsOfReview', 'KMeans_Cluster']].head())

# --- Business Value ---
print("\n=== Business Value ===")
print("1. Сегментація ринку за мовою та кількістю відгуків допомагає видавцям:")
print("   - Розуміти, які мовні сегменти приносять більше відгуків та, можливо, є більш прибутковими.")
print("   - Оптимізувати маркетингові стратегії, спрямовуючи зусилля на сегменти з високою активністю.")
print("2. Групування за роками публікації дозволяє відслідковувати динаміку змін:")
print("   - Аналізувати, як змінюється популярність мов та відгуки з часом.")
print("   - Визначати тренди: чи зростає кількість книг у певних кластерах, що може вказувати на зростання попиту.")
print("3. Загалом, цей аналіз допомагає приймати стратегічні рішення щодо:")
print("   - Планування майбутніх випусків.")
print("   - Розподілу рекламного бюджету та ресурсів.")
print("   - Оптимізації продуктового портфеля відповідно до ринкових трендів.\n")
