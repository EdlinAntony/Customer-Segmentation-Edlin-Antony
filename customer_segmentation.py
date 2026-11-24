import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = {
    'Student': ['A', 'B', 'C', 'D', 'E', 'F'],
    'Math': [2, 3, 10, 11, 25, 24],
    'Science': [3, 4, 11, 12, 30, 29]
}
df = pd.DataFrame(data)
X = df[['Math', 'Science']]

wcss = []
for i in range(1, 7):
    kmeans = KMeans(n_clusters=i, init='random', max_iter=300, n_init=10, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 7), wcss, marker='o')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of clusters (K)')
plt.ylabel('WCSS')
plt.show()

kmeans = KMeans(n_clusters=3, init='random', max_iter=300, n_init=10, random_state=42)
df['Cluster'] = kmeans.fit_predict(X)

plt.scatter(X['Math'], X['Science'], c=df['Cluster'], cmap='viridis', s=100)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            s=200, c='red', label='Centroids', marker='X')
plt.title("Customer/Student Segmentation (K-Means)")
plt.xlabel("Math Score")
plt.ylabel("Science Score")
plt.legend()
plt.show()

print(df[['Student', 'Math', 'Science', 'Cluster']])
