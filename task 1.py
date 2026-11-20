import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

wine = load_wine()
X, y = wine.data, wine.target

print("Number of samples:", X.shape[0])
print("Number of features:", X.shape[1])
print("Number of classes:", len(wine.target_names))
print("Feature names:", wine.feature_names)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

models = {
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

print(f"\n{'Model':15s} | {'Accuracy'}")
print("-" * 28)
for name, model in models.items():
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"{name:15s} | {acc:.4f}")

k_values = range(1, 21)
accuracies = []
for k in k_values:
    acc = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train).score(X_test, y_test)
    accuracies.append(acc)

best_k = k_values[accuracies.index(max(accuracies))]
print("\nBest value of k:", best_k, "-> accuracy =", round(max(accuracies), 4))

plt.plot(k_values, accuracies, marker="o")
plt.xticks(k_values)
plt.xlabel("k")
plt.ylabel("Accuracy")
plt.title("KNN Accuracy vs k")
plt.show()
