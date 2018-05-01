import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction import DictVectorizer

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

def readData(filepath):
	featureList = ['Capital','Prefix1','Prefix2','Prefix3','Suffix1','Suffix2','Suffix3','Hyphen','isNumber','hasNumber']

	trainFile = open(filepath+'train.txt', 'r')
	testFile = open(filepath+'test.txt','r')
	df_train = pd.read_table(trainFile, delim_whitespace=True, names=('word','posTag','chunkTag'))
	df_train['prevTag'] = 'NONE'
	df_train['prevTag'].loc[1:] = df_train['posTag'].as_matrix()[:-1]
	indices_train = df_train[df_train['word']=='.'].index + 1
	indices_train = indices_train[:-1]
	df_train['prevTag'].iloc[indices_train] = 'NONE'
	#print indices_train
	#print df_train 
	trainFile.close()

	df_test = pd.read_table(testFile, delim_whitespace=True, names=('word','posTag','chunkTag'))
	df_test['prevTag'] = 'NONE'
	df_test['prevTag'].loc[1:] = df_test['posTag'].as_matrix()[:-1]
	indices_test = df_test[df_test['word']=='.'].index + 1
	indices_test = indices_test[:-1]
	df_test['prevTag'].iloc[indices_test] = 'NONE'
	#print indices_test
	#print df_test
	testFile.close()

	corpus = df_train['word'].unique()
	tags = df_train['posTag'].unique()	
	#for p in corpus : print p
	#print corpus.shape (19122 vocabulary size, including puctuations)
	#print tags.shape   (44 pos tags, including punctuations)

	#df_train[featureList] = pd.DataFrame(train_features.values.tolist(), index=df_train.index) 	
	#df_test[featureList] = pd.DataFrame(test_features.values.tolist(), index=df_test.index) 

	return df_train, df_test, corpus, tags

def getTrainandTestFeatures(df_train, df_test):
	train_features = df_train['word'].apply(extractFeatures)
	test_features = df_test['word'].apply(extractFeatures)
	return train_features, test_features, df_train['posTag'], df_test['posTag']

def extractFeatures(word):
	#Features considered: prefix, suffix, capitalization, startWord, endWord, hasNumbers, hasHyphen, isNumber ....  
	features = {'capitalized':word[0].isupper(), 
	'prefix1': word[0:2], 
	'prefix2': word[0:3], 
	'prefix3':word[0:4], 
	'suffix1':word[-2:], 
	'suffix2':word[-3:], 
	'suffix3':word[-4:],
	'hasHyphen':'-' in word,
	'isNumber':word.isdigit(), 
	'hasNumber':any(p.isdigit() for p in word)
	}
	return features

if __name__ == '__main__':
	df_train, df_test, corpus, labels = readData('../data/')
	X_train, X_test, y_train, y_test = getTrainandTestFeatures(df_train, df_test)
	clf = Pipeline([('vectorizer', DictVectorizer(sparse=False)),('classifier', LogisticRegression(solver='lbfgs',multi_class='multinomial'))])
	clf.fit(X_train[:50000], y_train[:50000])
	y_pred = clf.predict(X_test)
	print clf.classes_
	print accuracy_score(y_test, y_pred)
	#for x in df_train.as_matrix(): print x
