# Naive Bayes  
# Câu 1: (1.5 điểm) Cho tập dữ liệu Iris từ sklearn.datasets. Hãy chia tập dữ liệu thành 
# tập huấn luyện và tập kiểm tra với tỷ lệ 80:20. Sử dụng thuật toán Naive Bayes để huấn 
# luyện mô hình và dự đoán trên tập kiểm tra và inn ra độ chính xác của mô hình. 

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

data = load_iris()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = GaussianNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Độ chính xác của mô hình:", accuracy)
