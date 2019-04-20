import pandas as pd
import numpy as np

import string
import pickle

from utils import readData
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import DictVectorizer


class MEMM:
	

	def __init__(self, mode='test'):
		self.mode = mode
		self.modelFileMaxEnt = 'maxEnt.sav'
		self.modelFileMaxEntPair = 'maxEntTagPairs.sav'

		if (self.mode == 'train'):
			self.MaxEntClassifier = None
			self.TwoLabelClassifier = None
			self.tags = None
			self.tagPairs = None
		elif (self.mode == 'test'):
			self.MaxEntClassifier = pickle.load(open(self.modelFileMaxEnt, 'rb'))
			self.TwoLabelClassifier = pickle.load(open(self.modelFileMaxEntPair, 'rb'))
			self.tags = self.MaxEntClassifier.classes_
			self.tagPairs = self.TwoLabelClassifier.classes_

		else:
			print('Invalid mode')
			exit(0)

	def preprocess(self, df_train, df_test):
		'''
		-Add field previous word and previous tag. 
		-Previous tag and current tag are required for the labels of the Maximum Entropy Classifier
		-Previous word is used as a feature
		-Extract dictionary of features using the method extract features
		'''	
		numOfTrainingSamples = df_train.shape[0]
		numOfTestSamples = df_test.shape[0]
		
		seriesTrainTags = pd.Series(np.empty(numOfTrainingSamples,dtype=object),dtype=object,name='prevTag')
		seriesTestTags = pd.Series(np.empty(numOfTestSamples,dtype=object),dtype=object,name='prevTag')
		seriesTrainPWords = pd.Series(np.empty(numOfTrainingSamples,dtype=object),dtype=object,name='prevWord')
		seriesTestPWords = pd.Series(np.empty(numOfTestSamples,dtype=object),dtype=object,name='prevWord')
		
		df_train = pd.concat([df_train, seriesTrainTags, seriesTrainPWords],axis=1)
		df_test = pd.concat([df_test, seriesTestTags, seriesTestPWords],axis=1)
		

		df_train['prevTag'] = df_train['posTag'].shift(1)
		df_test['prevTag'] = df_test['posTag'].shift(1)
		df_train['prevWord'] = df_train['word'].shift(1)
		df_test['prevWord'] = df_test['word'].shift(1)
		

		df_train['prevTag'].fillna('.',inplace=True)
		df_test['prevTag'].fillna('.',inplace=True)
		df_train['prevWord'].fillna('.',inplace=True)
		df_test['prevWord'].fillna('.',inplace=True)

		trainSeries = pd.Series(np.arange(numOfTrainingSamples))
		testSeries = pd.Series(np.arange(numOfTestSamples))
		train_features = trainSeries.apply(lambda x : self.extractFeatures(x, df_train))
		test_features = testSeries.apply(lambda x : self.extractFeatures(x, df_test))
		
		return train_features, test_features, df_train['posTag'], df_test['posTag'], df_train['prevTag']+' '+df_train['posTag'], df_test['prevTag']+' '+df_test['posTag']


	def extractFeatures(self, index, df):
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

	def viterbiDecoding(self, X):
		''' 
		Here X is a dataframe of a single sentence
		Number of rows in data frame is length of the sequence
		'''
		seqLen = X.shape[0]
		seriesPWords = pd.Series(np.empty(seqLen,dtype=object),dtype=object,name='prevWord')
		X = pd.concat([X,seriesPWords],axis=1)
		X['prevWord'] = X['word'].shift(1)
		a = pd.Series(np.arange(seqLen)).apply(lambda x: self.extractFeatures(x,X))
	
		Q = np.zeros((len(self.tags),seqLen))
		T = np.empty_like(Q, dtype=object)
		Q[:,0] = self.MaxEntClassifier.predict_proba(a[0])
		T[:,0] = self.tags
		probNext = self.TwoLabelClassifier.predict_proba(a)

		for w in np.arange(seqLen)[1:]:
			for (cTag,i) in zip(self.tags,np.arange(len(self.tags))):
				maxji = 0 
				tji = None
				for j in self.tags:
					Tji = 0
					tagPair = j+' '+cTag
					x =  np.where(self.tagPairs == tagPair)
					indList = x[0]
					if (indList.size != 0 ):
						currProb = Q[i,w-1] * probNext[w,indList[0]] 
						if (currProb > maxji):
							maxji = currProb
							tji = j
				Q[i,w] = maxji
				T[i,w] = tji
		
		tagList = np.empty(seqLen, dtype=object)
		lastTagInd = np.argmax(Q[:,seqLen-1])
		tagList[seqLen-1] = self.tags[lastTagInd]
		prevTag = T[lastTagInd, seqLen-1]
		for i in np.arange(seqLen)[1:]:
			j = seqLen-i-1
			tagList[j] = prevTag
			prevTagInd = np.where(self.tags == prevTag)[0][0]
			prevTag = T[prevTagInd, j]
		print ("--------------------Predicted Tags------------------------------------")
		print zip(X['word'].tolist(), tagList) 
		print ("----------------------------Actual Tags-----------------------------------")
		print zip(X['word'].tolist(), X['posTag'].tolist()) 


	def fit(self, X, y1, y2):
		if (self.mode == 'train'):
			print 'Training model for single tags'
			clf = Pipeline([('vectorizer', DictVectorizer(sparse=False)),('classifier', LogisticRegression(solver='lbfgs',multi_class='multinomial'))])
			clf.fit(X, y1)
			self.MaxEntClassifier = clf
			filename1 = self.modelFileMaxEnt
			pickle.dump(clf, open(filename1, 'wb'))
			
			print 'Training model for pairs of tags'
			clf2 = Pipeline([('vectorizer', DictVectorizer(sparse=False)),('classifier', LogisticRegression(solver='lbfgs',multi_class='multinomial'))])
			clf2.fit(X, y2)
			self.TwoLabelClassifier = clf2
			filename2 = self.modelFileMaxEntPair
			pickle.dump(clf2, open(filename2, 'wb'))
		
			self.tags = self.MaxEntClassifier.classes_
			self.tagPairs = self.TwoLabelClassifier.classes_
		else:
			print('Cannot fit in test mode')
			exit(0)


if __name__ == '__main__':
	df_train, df_test, corpus, tags = readData('../data/')
	posTagger = MEMM()
	X_train, X_test, y_train1, y_test1, y_train2, y_test2 = posTagger.preprocess(df_train[:50000], df_test[:50000])
	#print("Fitting model")
	#posTagger.fit(X_train, y_train1, y_train2)
	print ("Sample Tags using Viterbi decoding")
	posTagger.viterbiDecoding(df_test[:46])

