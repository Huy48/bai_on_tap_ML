# Cây quyết định 
# Câu 5: (2 điểm) Cho tập dữ liệu Breast Cancer từ sklearn.datasets. Huấn luyện mô hình 
# cây quyết định và đánh giá độ chính xác của mô hình trên tập kiểm tra (chia 75:25). 
# Xuất ra hình ảnh của cây quyết định.

import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score


data = load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

tree_model = DecisionTreeClassifier(random_state=42)
tree_model.fit(X_train, y_train)

y_pred = tree_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Độ chính xác của mô hình cây quyết định:", accuracy)

plt.figure(figsize=(20, 10))
plot_tree(tree_model, feature_names=data.feature_names, class_names=data.target_names, filled=True)
plt.title("Decision Tree for Breast Cancer Dataset")
plt.show()
