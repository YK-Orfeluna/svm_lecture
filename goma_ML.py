#!/usr/bin/env python
# -*- coding: utf-8 -*

import numpy as np

from sklearn import datasets

from sklearn.svm import SVC

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from sklearn.model_selection import GridSearchCV

from sklearn.decomposition import PCA

from sklearn.calibration import CalibratedClassifierCV

#import matplotlib.pyplot as plt


"""
if 1 :
	こっちを実行
else :
	こっちを実行したいなら，"if 0"にする
"""

if 1 :	#　irisデータ読み込み
	iris = datasets.load_iris()

	# 学習データ（train）とテストデータ(test)を作成	
	# xは特徴量，yはラベル（正解値）
	# 奇数番目と偶数番目で学習データとテストデータに分割 by 山添先生コード
	x_train = iris.data[xrange(0, len(iris.data), 2)]
	y_train = iris.target[xrange(0, len(iris.data), 2)]

	x_test = iris.data[xrange(1, len(iris.data), 2)]
	y_test = iris.target[xrange(1, len(iris.data), 2)]

	target_names = iris.target_names

else :	# 指定のcsvファイルを読み込む
	#　横に次元数のデータ+ラベル，となるようにcsvを作成する
	# ディレクトリ名に日本語が入っているとエラーの可能性
	#　学習データ
	data = np.genfromtxt("learn.csv", delimiter=',')

	# データファイルの横軸・縦軸要素数
	cols = len(data[0])
	rows = len(data)
	dim = cols-1

	X_learn = data[:,:dim]    
	y_learn = data[:,dim]

	# テストデータ
	data = np.genfromtxt("test.csv", delimiter=',')

	# データファイルの横軸・縦軸要素数
	cols = len(data[0])
	rows = len(data)
	dim = cols-1

	X_test = data[:,:dim]    
	y_test = data[:,dim]

# 学習の用意
svm = SVC(probability=True, decision_function_shape="ovr")

# 分類器の設定を確認する
params = svm.get_params()
print("Parameters = %s" %params)

# 学習データを学習させる
svm.fit(x_train, y_train)

#分類器にテストデータを分類させる
predict = svm.predict(x_test)
print("Predict = %s" %predict)

if 0 :	# F値を計算する
	# 混合行列を計算する
	# 配列横，ラベル順に予測されたもの（予測結果1→予測結果2→予測結果3）
	# 配列縦，順に正解ラベル
	matrix = confusion_matrix(y_test, predict)
	print("Confusion Matrix = %s" %matrix)

	#F値
	report = classification_report(y_test, predict, target_names=target_names)
	print(report)

else :	# グリッドサーチと交差検定
	# グリッドサーチするパラメータの選択肢を決定する
	para_svm = {"kernel":["linear", "rbf"], "C":[1.0, 2.0, 3.0]}
	# cvは交差検定回数
	svm = GridSearchCV(svm, para_svm, cv=10)

	if 1 :	#　交差検定をやるくらいだから（学習データとテストデータに分ける余裕がない），普通はこっち
		svm.fit(iris.data, iris.target)

		# 最高平均正解率とその時のパラメータ
		best_score = svm.best_score_
		print("Best score = %s" %best_score)
		best_param = svm.best_estimator_
		print("Best Parameter = %s" %best_param)

		#各パラメータの平均正解率とその標準偏差
		each_score = svm.grid_scores_
		print("Each socores = %s" %each_score)

	else :	#十分に学習データがあって，テストデータを用意できる場合，F値による精度検証も可能
		svm.fit(x_train, y_train)

		predict= svm.predict(x_test)

		best_score = svm.best_score_
		print("Best score = %s" %best_score)
		best_param = svm.best_estimator_
		print("Best Parameter = %s" %best_param)

		matrix = confusion_matrix(y_test, predict)
		print("Confusion Matrix = %s" %matrix)

		report = classification_report(y_test, predict, target_names=target_names)
		print(report)

# 主成分分析(PCA)で次元削減を行う
num = [i+1 for i in xrange(iris.data.shape[1])]

for i in num :	# 次元の数for文でループ
	iris = datasets.load_iris()

	#　削減後の次元数
	print("n_components = %d" %i)
	pca = PCA(n_components=i)

	#　主成分分析を行って，特徴量が推定結果に及ぼす影響を調査する
	pca.fit(iris.data)
	print("特徴寄与率 = %s" %pca.explained_variance_ratio_)

	# 次元削減
	pca_data = pca.transform(iris.data)
	# 次元削減した結果に応じてxデータを作り変える
	x_train = pca_data[xrange(0, len(iris.data), 2)]
	x_test = pca_data[xrange(1, len(iris.data), 2)]

	# 次元削減した特徴で学習
	svm = SVC(probability=True, decision_function_shape="ovr")
	svm.fit(x_train, y_train)

	predict = svm.predict(x_test)

	matrix = confusion_matrix(y_test, predict)
	print("Confusion Matrix = %s" %matrix)

	report = classification_report(y_test, predict, target_names=target_names)
	print(report)

# 事後確率を求める
svm = SVC(probability=True, decision_function_shape="ovr")
svm.fit(x_train, y_train)

#スコア（予測確率：predict probability）を取得する
#各配列内，順に各ラベルに対する予測確率となる
score = svm.predict_proba(x_test)
print("Predict Probability = %s" %score)

if 0 :	# sigmoidでキャリブレーション
	sigmoid = CalibratedClassifierCV(svm, cv=2, method='sigmoid')
	sigmoid.fit(x_train, y_train)
	print("Sigmoid_Ppsterior Probability = %s" %sigmoid.predict_proba(x_test))
	"""
	plt.figure()
	print score
	s1 = score[:25, 0]
	s2 = score[25:50, 1]
	s3 = score[50:, 2]
	plt.bar([i for i in xrange(len(s1))], s1, 0.5, label=iris.target_names[0], color="0.8", align="center")
	plt.bar([i+len(s1) for i in xrange(len(s2))], s2, 0.5, label=iris.target_names[1], color="0.6", align="center")
	plt.bar([i+len(s1)+len(s2)for i in xrange(len(s3))], s2, 0.5, label=iris.target_names[2], color="0.4", align="center")
	plt.legend(loc=4)
	plt.ylim(0.0, 1.0)
	plt.xlim(1, 75)
	plt.title("Predict Probability")
	plt.show()

	plt.figure()
	pp = sigmoid.predict_proba(x_test)
	arr1 = pp[:25, 0]
	arr2 = pp[25:50, 1]
	arr3 = pp[50:, 2]
	plt.bar([i for i in xrange(len(arr1))], arr1, 0.5, label=iris.target_names[0], color="0.8", align="center")
	plt.bar([i+len(arr1) for i in xrange(len(arr2))], arr2, 0.5, label=iris.target_names[1], color="0.6", align="center")
	plt.bar([i+len(arr1)+len(arr2)for i in xrange(len(arr3))], arr3, 0.5, label=iris.target_names[2], color="0.4", align="center")
	plt.legend(loc=4)
	plt.ylim(0.0, 1.0)
	plt.xlim(1, 75)
	plt.title("Posterior Probability with sigmoid")
	plt.show()
	"""
else :	# isotonicでキャリブレーション
	isotonic = CalibratedClassifierCV(svm, cv=2, method='isotonic')
	isotonic.fit(x_train, y_train)
	print("isotonic_Posterior Probability = %s" %isotonic.predict_proba(x_test))
	"""
	plt.figure()
	pp = isotonic.predict_proba(x_test)
	arr1 = pp[:25, 0]
	arr2 = pp[25:50, 1]
	arr3 = pp[50:, 2]
	plt.bar([i for i in xrange(len(arr1))], arr1, 0.5, label=iris.target_names[0], color="0.8", align="center")
	plt.bar([i+len(arr1) for i in xrange(len(arr2))], arr2, 0.5, label=iris.target_names[1], color="0.6", align="center")
	plt.bar([i+len(arr1)+len(arr2)for i in xrange(len(arr3))], arr3, 0.5, label=iris.target_names[2], color="0.4", align="center")
	plt.legend(loc=4)
	plt.ylim(0.0, 1.0)
	plt.xlim(1, 75)
	plt.title("Posterior Probability with isotonic")
	plt.show()
	"""
if 0 :	# 外部ファイルへの書き出し方法
	from sklearn.externals import joblib
	joblib.dump(svm, 'svm.pkl') 

	# 読み込み
	# joblib.load('svm.pkl') 
