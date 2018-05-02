import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction import DictVectorizer

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

def readData(filepath):
	trainFile = open(filepath+'train.txt', 'r')
	testFile = open(filepath+'test.txt','r')
	df_train = pd.read_table(trainFile, delim_whitespace=True, names=('word','posTag','chunkTag'))
	trainFile.close()

	df_test = pd.read_table(testFile, delim_whitespace=True, names=('word','posTag','chunkTag'))
	testFile.close()

	corpus = df_train['word'].unique()
	tags = df_train['posTag'].unique()	
	#for p in corpus : print p
	#print corpus.shape (19122 vocabulary size, including puctuations)
	#print tags.shape   (44 pos tags, including punctuations)

	return df_train, df_test, corpus, tags

def getTrainandTestData(df_train, df_test):
	df_train['prevTag'] = '.'
	df_train['prevTag'].loc[1:] = df_train['posTag'].as_matrix()[:-1]
#	indices_train = df_train[df_train['word']=='.'].index + 1
#	indices_train = indices_train[:-1]
#	df_train['prevTag'].iloc[indices_train] = 'NONE'
	#print indices_train
	#print df_train 
	
	df_train['prevWord'] = '.'
	df_train['prevWord'].loc[1:] = df_train['word'].as_matrix()[:-1]
#	ind1 = df_train[df_train['word']=='.'].index+1
#	ind1 = ind1[:-1]
#	df_train['prevWord'].iloc[ind1] = 'NONE'

	df_test['prevTag'] = '.'
	df_test['prevTag'].loc[1:] = df_test['posTag'].as_matrix()[:-1]
#	indices_test = df_test[df_test['word']=='.'].index + 1
#	indices_test = indices_test[:-1]
#	df_test['prevTag'].iloc[indices_test] = 'NONE'
	#print indices_test
	#print df_test
	
	df_test['prevWord'] = '.'
	df_test['prevWord'].loc[1:] = df_test['word'].as_matrix()[:-1]
#	ind2 = df_test[df_test['word']=='.'].index+1
#	ind2 = ind2[:-1]
#	df_test['prevWord'].iloc[ind2] = 'NONE'

	df_train['twoTag'] = df_train['prevTag']+df_train['posTag']
	df_test['twoTag'] = df_test['prevTag']+df_test['posTag']

	numOfTrainingSamples = df_train.shape[0]
	numOfTestSamples = df_test.shape[0]
	trainSeries = pd.Series(np.arange(numOfTrainingSamples))
	testSeries = pd.Series(np.arange(numOfTestSamples))
	train_features = trainSeries.apply(lambda x : extractFeatures(x, df_train))
	test_features = testSeries.apply(lambda x : extractFeatures(x, df_test))
	return train_features, test_features, df_train['posTag'], df_test['posTag'], df_train['twoTag'], df_test['twoTag']

def extractFeatures(index, df):
	#Features considered: isCapitalized, Prefixes of length up to 3, Suffixes of length up to 3, hasHyphen, isNumber, hasNumber, is the starting word and previous word 
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

if __name__ == '__main__':
	df_train, df_test, corpus, labels = readData('../data/')
	#print df_train
	X_train, X_test, y_train1, y_test1, y_train2, y_test2 = getTrainandTestData(df_train, df_test)
	
	#mlb = MultiLabelBinarizer()
	#y_train2tag = mlb.fit_transform(y_train2)
	clf1 = Pipeline([('vectorizer', DictVectorizer(sparse=False)),('classifier', LogisticRegression(solver='lbfgs',multi_class='multinomial',verbose=1))])
	clf1.fit(X_train[:10000], y_train2[:10000])

	y_pred = clf1.predict(X_test)
	print clf1.classes_
	print accuracy_score(y_test2, y_pred)
	#for x in df_train[:10]: print x
