#!/usr/bin/env python
# -*- coding: utf-8 -*

import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.calibration import CalibratedClassifierCV

if 1 :

	x1 = np.random.rand(100, 2)
	x1[:, 0] = x1[:, 0] * 10
	x1[:, 1] = x1[:, 1] * 0.5
	x2 = np.random.rand(100, 2)
	x2[:, 0] = x2[:, 0] * 10 + 100
	x2[:, 1] = x2[:, 1] * 0.5
	print x2
	x_train = np.append(x1, x2, axis=0)

else :
	x1 = np.random.rand(100, 2)
	x1[:, 0] = x1[:, 0] * 50
	x1[:, 1] = x1[:, 1] * 0.5
	x2 = np.random.rand(100, 2)
	x2[:, 0] = x2[:, 0] * 50 + 50
	x2[:, 1] = x2[:, 1] * 0.5
	print x2
	x_train = np.append(x1, x2, axis=0)

y1 = np.zeros(100)
y2 = np.ones(100)
y_train = np.append(y1, y2)

x_test = np.array([[0, 0.25], [10, 0.25], [20, 0.25], [30, 0.25], [40, 0.25], [50, 0.25], [60, 0.25], [70, 0.25], [80, 0.25], [90, 0.25], [100, 0.25], [110, 0.25]])
y_test = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])



plt.figure()
plt.scatter(x_train[:100, 0], x_train[:100, 1], c="red", label="train_A")
plt.scatter(x_train[100:, 0], x_train[100:, 1], c="blue", label="train_B")
plt.xlim(-0.9, 110.9)
plt.ylim(-1, 1.5)
plt.legend()
plt.title("Train Data")
plt.show()

plt.figure()
plt.scatter(x_train[:100, 0], x_train[:100, 1], c="red", label="train_A")
plt.scatter(x_train[100:, 0], x_train[100:, 1], c="blue", label="train_B")
plt.scatter(x_test[:, 0], x_test[:, 1], s=50, c="green", label="test")
plt.xlim(-0.9, 110.9)
plt.ylim(-1, 1.5)
plt.title("Train & Test Data")
plt.legend()
plt.show()

svm = SVC(C=3.0, probability=True, decision_function_shape="ovr", kernel="linear")
svm.fit(x_train, y_train)
predict = svm.predict(x_test)
print(predict)
score = svm.predict_proba(x_test)
print type(score)
report = classification_report(y_test, predict)
print(report)

plt.figure()
plt.plot(x_test[:, 0], score[:, 0], label="A", c="red")
plt.plot(x_test[:, 0], score[:, 1], label="B", c="blue")
plt.ylim(0, 1.1)
plt.legend()
plt.title("Normal SVM")
plt.ylabel("Preditc Probability")
plt.xlabel("x-coordinate")
plt.show()

sigmoid = CalibratedClassifierCV(svm, cv=2, method='sigmoid')
sigmoid.fit(x_train, y_train)
predict = sigmoid.predict(x_test)
pp = sigmoid.predict_proba(x_test)
print("Sigmoid_Posterior Probability = %s" %pp)
report = classification_report(y_test, predict)
print(report)

plt.figure()
plt.plot(x_test[:, 0], pp[:, 0], label="A", c="red")
plt.plot(x_test[:, 0], pp[:, 1], label="B", c="blue")
plt.ylim(0, 1.1)
plt.legend()
plt.title("Sigmoid Calibration")
plt.ylabel("Posterior Probability")
plt.xlabel("x-coordinate")
plt.show()

isotonic = CalibratedClassifierCV(svm, cv=2, method='isotonic')
isotonic.fit(x_train, y_train)
predict = isotonic.predict(x_test)
pp = isotonic.predict_proba(x_test)
print("isotonic_Posterior Probability = %s" %pp)
report = classification_report(y_test, predict)
print(report)

plt.figure()
plt.plot(x_test[:, 0], pp[:, 0], label="A", c="red")
plt.plot(x_test[:, 0], pp[:, 1], label="B", c="blue")
plt.ylim(0, 1.1)
plt.legend()
plt.title("Isotonic Calibration")
plt.ylabel("Posterior Probability")
plt.xlabel("x-coordinate")
plt.show()

exit()
