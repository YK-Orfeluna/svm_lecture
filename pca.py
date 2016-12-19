# -*- coding: utf-8 -*
"""
次元削減
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix

from sklearn.calibration import CalibratedClassifierCV

from sklearn.decomposition import PCA
from sklearn.metrics import (precision_score, recall_score, f1_score)


iris = datasets.load_iris()
num = [i+1 for i in xrange(iris.data.shape[1])]
#num=[4]

for i in num :
	print("n_components = %d" %i)

	iris = datasets.load_iris()

	pca = PCA(n_components=i)
	pca.fit(iris.data)
	print("特徴寄与率 = %s" %pca.explained_variance_ratio_)
	iris.data = pca.transform(iris.data)


	x_train = iris.data[xrange(0, len(iris.data), 2)]
	y_train = iris.target[xrange(0, len(iris.data), 2)]

	x_test = iris.data[xrange(1, len(iris.data), 2)]
	y_test = iris.target[xrange(1, len(iris.data), 2)]

	x_all = iris.data
	y_all = iris.target

	svm = SVC(probability=True, decision_function_shape="ovr")
	svm.fit(x_train, y_train)

	#print(svm.predict(x_test))
	#print(svm.predict_proba(x_test))
	#print(np.sum(svm.predict_proba(x_test), axis=1))
	#print(confusion_matrix(y_test, svm.predict(x_test)))

	print(confusion_matrix(y_test, svm.predict(x_test)))

	print("P = %1.3f" %precision_score(y_test, svm.predict(x_test)))
	print("R = %1.3f" %recall_score(y_test, svm.predict(x_test)))
	print("F1 = %1.3f" %f1_score(y_test, svm.predict(x_test)))
