# SVM 
# Câu 6: (1 điểm) Dùng tập dữ liệu Digits từ sklearn.datasets. Huấn luyện mô hình SVM 
# với kernel tuyến tính (linear). Đánh giá độ chính xác trên tập kiểm tra. 

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

data = load_digits()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

y_pred = svm_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Độ chính xác của mô hình SVM với kernel tuyến tính:", accuracy)
