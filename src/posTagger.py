import pandas as pd
import numpy as np

import string

from utils import readData
from sklearn.linear_model import LogistricRegression

class MEMM:

	def __init__(self):
		self.MaxEntClassifier = None
		self.TwoLabelClassifier = None

	def preprocess(df_train, df_test):
		'''
		-Add field previous word and previous tag. 
		-Previous tag and current tag are required for the labels of the Maximum Entropy Classifier
		-Previous word is used as a feature
		-Extract dictionary of features using the method extract features
		'''	

		df_train['prevTag'] = '.'
		df_train['prevTag'].loc[1:] = df_train['posTag'].as_matrix()[:-1]
	
		df_train['prevWord'] = '.'
		df_train['prevWord'].loc[1:] = df_train['word'].as_matrix()[:-1]

		df_test['prevTag'] = '.'
		df_test['prevTag'].loc[1:] = df_test['posTag'].as_matrix()[:-1]
	
		df_test['prevWord'] = '.'
		df_test['prevWord'].loc[1:] = df_test['word'].as_matrix()[:-1]

		numOfTrainingSamples = df_train.shape[0]
		numOfTestSamples = df_test.shape[0]
		trainSeries = pd.Series(np.arange(numOfTrainingSamples))
		testSeries = pd.Series(np.arange(numOfTestSamples))
		train_features = trainSeries.apply(lambda x : extractFeatures(x, df_train))
		test_features = testSeries.apply(lambda x : extractFeatures(x, df_test))
		return train_features, test_features, df_train['posTag'], df_test['posTag'], df_train['prevTag'], df_test['prevTag']


	def extractFeatures(index, df):
		'''
		-Extract features for each word in a sequence
		-Following features are extracted:
			> is capitalized?
			> prefixes of length 2, 3 and 4
			> suffixes of length 2, 3 and 4
			> has hyphen?
			> is a numeric quantity?
			> has a number within it?
			> Is it the start of the sequence?
			> previous Word 
		'''

		word = df['word'].loc[index]
		prevWord = df['prevWord'].loc[index]
		features = {'capitalized':word[0].isupper(), 
		'prefix1': word[0:2], 
		'prefix2': word[0:3], 
		'prefix3':word[0:4], 
		'suffix1':word[-2:],
		'suffix2':word[-3:], 
		'suffix3':word[-4:],
		'hasHyphen':'-' in word,
		'isNumber':word.isdigit(), 
		'hasNumber':any(p.isdigit() for p in word),
		'isStart': prevWord == '.',
		'prevWord': prevWord
		}
		return features

	def viterbiDecoding(X, y):
		''' 
		Here X is a dataframe of a single sentence
		Number of rows in data frame is length of the sequence
		'''
		seqLen = X.shape[0]
		self.TwoLabelClassifier.predict_proba(X)		
	
		Q = np.zeros((self.numTags,seqLen))
		Q[0,:] = self.MaxEntClassifierpredict_proba(X)
		Q[1,:] = np.maximum()

	def fit(X, y):
		clf = Pipeline([('vectorizer', DictVectorizer(sparse=False)),('classifier', LogisticRegression(solver='lbfgs',multi_class='multinomial'))])
		clf.fit(X, y[:,1])
		self.MaxEntClassifier = clf

		clf2 = Pipeline([('vectorizer', DictVectorizer(sparse=False)),('classifier', LogisticRegression(solver='lbfgs',multi_class='multinomial'))])
		clf2.fit(X, y[:,2])
		self.TwoLabelClassifier = clf2

	def predict(x):
	
	
	def predict_all(X):


if __name__ == '__main__':
	df_train, df_test, corpus, tags = readData('../data/')
	X_train, X_test, y_train 
