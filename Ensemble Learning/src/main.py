import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn import tree
from sklearn import linear_model
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
import random
import math

class Bagging:
	def __init__(self):
		self.classifiers = []
	def name(self):
		return "bagging"
	def train(self, x, y, base_classifier):
		# bootstrap
		sampleX, sampleY = [], []
		for i in range(len(x)):
			randIndex = random.randint(0, len(x) - 1)
			sampleX.append(x[randIndex].copy().tolist())
			sampleY.append(y[randIndex].copy())
		sampleX = np.array(sampleX)
		sampleY = np.array(sampleY)
		cl = base_classifier.newInstance()
		cl.train(sampleX, sampleY)
		self.classifiers.append(cl)
	def predict(self, x):
		positive, negative = 0, 0
		for classifier in self.classifiers:
			if classifier.predict(x) == 1:
				positive += 1
			else:
				negative += 1
		if positive > negative:
			return 1
		else:
			return 0

class AdaBoostM1:
	def name(self):
		return "AdaBoostM1"
	def __init__(self, N): # N: sample_num
		self.weight = [1.0 / N for i in range(N)]
		self.voting = []
		self.classifiers = []
	def train(self, x, y, base_classifier):
		cl = base_classifier.newInstance()
		cl.train(x, y, self.weight)
		Et = 0.0
		for i in range(len(x)):
			if not(cl.predict(x[i]) == y[i]):
				Et += self.weight[i]
		beta = Et / (1.0 - Et)
		for i in range(len(x)):
			if cl.predict(x[i]) == y[i]:
				self.weight[i] *= beta
		# normalize
		sum_weight = sum(self.weight)
		if sum_weight > 0.0:
			self.weight = [w / sum_weight for w in self.weight]
		else:
			self.weight = [1.0 / len(self.weight) for w in self.weight]
		if beta > 0.0:
			tmp = math.log(1.0 / beta)
		else:
			tmp = math.log(1e7)
		self.voting.append(tmp)
		self.classifiers.append(cl)
	def predict(self, x):
		positive, negative = 0, 0
		for i in range(len(self.classifiers)):
			if self.classifiers[i].predict(x) == 0:
				negative += self.voting[i]
			else:
				positive += self.voting[i]
		if positive > negative:
			return 1
		else:
			return 0

class DecisionTree:
	def name(self):
		return "decisionTree"
	def newInstance(self):
		return DecisionTree()
	def train(self, x, y, weight = None):
		self.classifier = tree.DecisionTreeClassifier()
		self.classifier.fit(x, y, sample_weight = weight)
	def predict(self, x):
		return self.classifier.predict(x)[0]

class SVM:
	def name(self):
		return "svm"
	def newInstance(self):
		return SVM()
	def train(self, x, y, weight = None):
		self.classifier = sklearn.svm.SVC(kernel = 'linear')
		self.classifier.fit(x, y, sample_weight = weight)
	def predict(self, x):
		return self.classifier.predict(x)[0]

class NaiveBayes:
	def name(self):
		return "NaiveBayes"
	def newInstance(self):
		return NaiveBayes()
	def train(self, x, y, weight = None):
		self.classifier = BernoulliNB()
		self.classifier.fit(x, y, sample_weight = weight)
	def predict(self, x):
		return self.classifier.predict(x)[0]

# # no balance
# def sample(train_rate, x, y):
#     trainX, trainY, testX, testY = [], [], [], []
#     # Simple Random Sample (jian dan sui ji chou yang)
#     for i in range(len(x)):
#         if random.random() < train_rate:
#             trainX.append(x[i].copy().tolist())
#             trainY.append(y[i].copy())
#         else:
#             testX.append(x[i].copy().tolist())
#             testY.append(y[i].copy())
#     trainX = np.array(trainX)
#     trainY = np.array(trainY)
#     testX = np.array(testX)
#     testY = np.array(testY)
#     return trainX, trainY, testX, testY

def balancedSample(train_rate, x, y):
	# 1803 spam hosts : y = 1 : positive
	# 4411 normal hosts : y = 0 : negative
	trainX, trainY, testX, testY = [], [], [], []
	# Stratrified Sampling (fen ceng chou yang / lei xing chou yang)
	df = pd.DataFrame(x) 
	positiveSample = df[y == 1].values
	negativeSample = df[y == 0].values
	for i in range(len(negativeSample)):
		if random.random() < train_rate:
			trainX.append(negativeSample[i].copy().tolist())
			trainY.append(0)
		else:
			testX.append(negativeSample[i].copy().tolist())
			testY.append(0)

	# 1803 spam hosts : y = 1 : positive
	# 4411 normal hosts : y = 0 : negative
	# so need more spam hosts: trainPositiveForMultiply: by copy (random)
	trainPositiveForMultiply = []
	for i in range(len(positiveSample)):
		if random.random() < train_rate:
			trainX.append(positiveSample[i].copy().tolist())
			trainY.append(1)
			trainPositiveForMultiply.append(positiveSample[i].copy().tolist())
		else:
			testX.append(positiveSample[i].copy().tolist())
			testY.append(1)

	# Multiply positive training samples (duo jie duan chou yang)
	# so need more spam hosts: trainPositiveForMultiply: by copy (random)
	deltaForBalance = int((len(negativeSample) - len(positiveSample)) * train_rate)
	for i in range(deltaForBalance):
		randIndex = random.randint(0, len(trainPositiveForMultiply) - 1)
		trainX.append(trainPositiveForMultiply[randIndex])
		trainY.append(1)

	trainX = np.array(trainX)
	trainY = np.array(trainY)

	testX = np.array(testX)
	testY = np.array(testY)

	return trainX, trainY, testX, testY


def preprocessing(onlyContent, onlyLink, normalize):
	# read_csv
	df = pd.read_csv('./ContentNewLinkAllSample.csv')
	# if spam: y = 1; if normal: y = 0
	df['class'] = df['class'].map(lambda c: 1 if c == 'spam' else 0)
	y = df['class']
	# get x
	if onlyContent:
		x = df[df.columns[:-139]].values
	elif onlyLink:
		x = df[df.columns[-138:-1]].values
	else:
		x = df[df.columns[:-1]].values
	if normalize:
		x = sklearn.preprocessing.normalize(x, axis = 0)
	return balancedSample(0.8, x, y)
	# return sample(0.8, x, y)
	
def test(testX, testY, classifier):
	positive_right, negative_right, positive_wrong, negative_wrong = 0, 0, 0, 0    
	for i in range(len(testX)):
		# predict & count (statistics: tong ji)
		prediction = classifier.predict(testX[i])
		if prediction == 1 and testY[i] == 1:
			positive_right += 1
		elif prediction == 0 and testY[i] == 0:
			negative_right += 1
		elif prediction == 1 and testY[i] == 0:
			positive_wrong += 1
		elif prediction == 0 and testY[i] == 1:
			negative_wrong += 1
	# calc result: accuracy, precision, recall, F1score
	accuracy = float(positive_right + negative_right) / len(testX)
	if positive_right + positive_wrong > 0.0:
		precision = float(positive_right) / ((positive_right + positive_wrong) * 1.0)
	else:
		precision = 0.0
	if positive_right + negative_wrong > 0.0:
		recall = float(positive_right) / ((positive_right + negative_wrong) * 1.0)
	else:
		recall = 0.0
	if precision + recall > 0:
		F1score = 2.0 * precision * recall / ((precision + recall) * 1.0)
	else:
		F1score = 0.0
	return accuracy, precision, recall, F1score

def ensemble(onlyContent = False, onlyLink = False, normalize = True, rounds = 50):
	trainX, trainY, testX, testY = preprocessing(onlyContent, onlyLink, normalize)
	# you can change ensemble_learninig_algorithm & base_classifier here
	# ensemble_learninig_algorithm: Bagging, AdaBoostM1
	# base_classifier: DecisionTree, SVM, NaiveBayes
	for ensemble_learninig_algorithm in [Bagging()]:
	# for ensemble_learninig_algorithm in [AdaBoostM1(len(trainY))]:
		for base_classifier in [DecisionTree()]:
		# for base_classifier in [SVM()]:
		# for base_classifier in [NaiveBayes()]:
			filename  = 'result/' + ensemble_learninig_algorithm.name() + "_" + base_classifier.name()
			if onlyContent:
				filename += "_onlyContent"
			elif onlyLink:
				filename += "_onlyLink"
			else:
				filename += "_contentAndLink"
			if normalize:
				filename += "_normalize"
			filename += '.csv'
			fout = open(filename, 'w')
			fout.write('round,accuracy,precision,recall,F1score\n')
			for i in range(0, rounds):
				print ensemble_learninig_algorithm.name(), base_classifier.name(), 'round', i
				ensemble_learninig_algorithm.train(trainX, trainY, base_classifier)
				result = test(testX, testY, ensemble_learninig_algorithm)
				fout.write(str(i+1) + ',')
				fout.write(str(result[0]) + ',' + str(result[1]) + ',' + str(result[2]) + ',' + str(result[3]) + '\n')
			fout.close()

if __name__ == '__main__':
	ensemble(onlyContent = False, onlyLink = False, normalize = True, rounds = 6)
	# ensemble(onlyContent = True, onlyLink = False, normalize = True, rounds = 6)
	# ensemble(onlyContent = False, onlyLink = True, normalize = True, rounds = 6)
	# ensemble(onlyContent = False, onlyLink = False, normalize = False, rounds = 6)
