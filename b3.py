# KNN 
# Câu 3: (1.25 điểm) Tải tập dữ liệu Wine từ sklearn.datasets và chia tập dữ liệu theo tỷ 
# lệ 70:30. Xây dựng mô hình KNN để phân loại dữ liệu. Sử dụng k = 5. Tính toán và in 
# ra độ chính xác, độ nhạy (recall), và độ chính xác (precision) của mô hình.

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

print("Độ chính xác:", accuracy)
print("Độ nhạy:", recall)
print("Độ chính xác:", precision)
