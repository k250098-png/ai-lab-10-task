import time
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

digits = load_digits()
X_scaled = StandardScaler().fit_transform(digits.data)

pca_2d = PCA(n_components=2)
X_2d = pca_2d.fit_transform(X_scaled)

for i, name in enumerate(digits.target_names):
    plt.scatter(X_2d[digits.target == i, 0], X_2d[digits.target == i, 1], label=name, s=20)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.show()

X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2, random_state=42, stratify=digits.target)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

pca_95 = PCA(n_components=0.95)
X_train_pca = pca_95.fit_transform(X_train_scaled)
X_test_pca = pca_95.transform(X_test_scaled)
print("Minimum components for 95% variance:", pca_95.n_components_)

t0 = time.time()
knn_orig = KNeighborsClassifier(n_neighbors=5).fit(X_train_scaled, y_train)
acc_orig = accuracy_score(y_test, knn_orig.predict(X_test_scaled))
time_orig = time.time() - t0

t0 = time.time()
knn_pca = KNeighborsClassifier(n_neighbors=5).fit(X_train_pca, y_train)
acc_pca = accuracy_score(y_test, knn_pca.predict(X_test_pca))
time_pca = time.time() - t0

print(f"Original (64 features)         -> Accuracy: {acc_orig:.4f}, Time: {time_orig:.4f}s")
print(f"PCA 95% ({pca_95.n_components_} features) -> Accuracy: {acc_pca:.4f}, Time: {time_pca:.4f}s")
