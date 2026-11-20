from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report

data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42, stratify=data.target)

tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, y_train)
y_pred = tree.predict(X_test)

def confusion(y_true, y_pred):
    TP = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    TN = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    FP = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    FN = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    return TP, TN, FP, FN

def precision(TP, FP):
    return TP / (TP + FP)

def recall(TP, FN):
    return TP / (TP + FN)

def f1(prec, rec):
    return 2 * prec * rec / (prec + rec)

TP, TN, FP, FN = confusion(y_test, y_pred)
accuracy = (TP + TN) / (TP + TN + FP + FN)
prec = precision(TP, FP)
rec = recall(TP, FN)
f1_val = f1(prec, rec)

print("TP =", TP, "TN =", TN, "FP =", FP, "FN =", FN)
print("Manual  ->", round(accuracy, 3), round(prec, 3), round(rec, 3), round(f1_val, 3))
print("\nConfusion matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification report:\n", classification_report(y_test, y_pred, target_names=data.target_names))
