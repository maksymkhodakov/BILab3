import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

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

# Візуалізація результатів кластеризації KMeans (1) - з використанням перших двох компонентів закодованих змінних
plt.figure(figsize=(8, 6))
plt.scatter(df[language_encoded_df.columns[0]], df['CountsOfReview'], c=df['KMeans_Cluster'], cmap='viridis', alpha=0.6)
plt.title('KMeans Clustering (Language vs CountsOfReview)')
plt.xlabel(language_encoded_df.columns[0])
plt.ylabel('CountsOfReview')
plt.colorbar(label='Cluster')
plt.show()

# Візуалізація результатів кластеризації KMeans з центрами кластерів
plt.figure(figsize=(8, 6))
plt.scatter(df[language_encoded_df.columns[0]], df['CountsOfReview'], c=df['KMeans_Cluster'], cmap='viridis', alpha=0.6)
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=200, label='Cluster Centers')
plt.title('KMeans Clustering with Centers (Language vs CountsOfReview)')
plt.xlabel(language_encoded_df.columns[0])
plt.ylabel('CountsOfReview')
plt.colorbar(label='Cluster')
plt.legend()
plt.show()

# Додатково: відображення перших кількох рядків даних з класифікацією
print("\nПерші кілька рядків з класифікацією:")
print(df[['Language', 'CountsOfReview', 'KMeans_Cluster']].head())
