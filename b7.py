# Câu 7: (1 điểm) Thử nghiệm với các kernel khác (rbf, poly). So sánh độ chính xác và 
# thời gian huấn luyện giữa các kernel này. Đưa ra nhận xét về kernel nào hiệu quả nhất 
# cho tập dữ liệu này. 

import time

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
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

kernels = ['linear', 'rbf', 'poly']
results = {}

for kernel in kernels:
	svm_model = SVC(kernel=kernel)
	
	start_time = time.time()
	svm_model.fit(X_train, y_train)
	train_time = time.time() - start_time

	y_pred = svm_model.predict(X_test)
	accuracy = accuracy_score(y_test, y_pred)

	results[kernel] = {'accuracy': accuracy, 'train_time': train_time}

for kernel, metrics in results.items():
	print(f"Kernel: {kernel}")
	print(f" - Độ chính xác: {metrics['accuracy']:.4f}")
	print(f" - Thời gian huấn luyện: {metrics['train_time']:.4f} giây")
	print("")

best_kernel = max(results, key=lambda k: (results[k]['accuracy'], -results[k]['train_time']))
print(f"Kernel hiệu quả nhất: {best_kernel}")
