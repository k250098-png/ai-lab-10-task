import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

X, _ = make_blobs(n_samples=200, centers=[[25, 20], [25, 80], [55, 50], [85, 20], [85, 80]],
                  cluster_std=6, random_state=3)

inertias = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    inertias.append(round(km.inertia_, 1))

print("Inertia for k = 1..10:", inertias)

plt.plot(range(1, 11), inertias, marker="o")
plt.xlabel("k")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.show()

kmeans = KMeans(n_clusters=5, n_init=10, random_state=42)
labels = kmeans.fit_predict(X)
print("Cluster centers:\n", kmeans.cluster_centers_.round(2))

plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", s=20)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c="red", marker="X", s=200)
plt.xlabel("Annual Income (k)")
plt.ylabel("Spending Score")
plt.title("K-Means Clustering (k = 5)")
plt.show()

for i, center in enumerate(kmeans.cluster_centers_):
    count = sum(labels == i)
    inc, score = center
    if inc < 40 and score < 40:
        name = "Low income, low spending"
    elif inc < 40 and score > 60:
        name = "Low income, high spending"
    elif inc > 70 and score < 40:
        name = "High income, low spending"
    elif inc > 70 and score > 60:
        name = "High income, high spending"
    else:
        name = "Average income, average spending"
    print(f"Cluster {i}: {count} customers | Centroid: {center.round(2)} | {name}")
