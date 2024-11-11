# Multilayer Perceptron 
# Câu 8: (2 điểm) Dùng tập dữ liệu MNIST. Huấn luyện một MLP với hai tầng ẩn (hidden 
# layers) để phân loại ảnh số từ 0-9. In ra độ chính xác trên tập kiểm tra. 

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

mnist = fetch_openml('mnist_784', version=1)
X = mnist.data
y = mnist.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

mlp_model = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=100, random_state=42)
mlp_model.fit(X_train, y_train)

y_pred = mlp_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Độ chính xác của mô hình MLP:", accuracy)
