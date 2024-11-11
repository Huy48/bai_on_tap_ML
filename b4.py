# Câu 4: (0.75 điểm) Thử nghiệm với các giá trị k khác nhau (1, 3, 7, 9). Vẽ đồ thị thể 
# hiện mối quan hệ giữa k và độ chính xác của mô hình. Nhận xét về kết quả. 


import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score

data = load_wine()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred, average='macro')
precision = precision_score(y_test, y_pred, average='macro')

k_values = [1, 3, 5, 7, 9]
accuracies = []


for k in k_values:
	knn = KNeighborsClassifier(n_neighbors=k)
	knn.fit(X_train, y_train)
	y_pred = knn.predict(X_test)
	accuracy = accuracy_score(y_test, y_pred)
	accuracies.append(accuracy)


plt.figure(figsize=(8, 6))
plt.plot(k_values, accuracies, marker='o', linestyle='-')
plt.xlabel('Giá trị của k')
plt.ylabel('Độ chính xác')
plt.title('Mối quan hệ giữa k và độ chính xác của mô hình KNN')
plt.show()

