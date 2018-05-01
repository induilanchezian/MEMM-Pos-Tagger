import numpy as np
import pandas as pd

def readDataset(filepath):
	trainFile = open(filepath, 'r')
	testFile = open(filepath,'r')
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
	for p in corpus : print p
	#print corpus.shape (19122 vocabulary size, including puctuations)
	#print tags.shape   (44 pos tags, including punctuations)

	return df_train, df_test, corpus, tags

def extractFeatures(word):
	#Features considered: prefix, suffix, capitalization, startWord, endWord, hasNumbers, hasHyphen, isNumber ....  
	features = np.array([0,'','','','','','',0,0,0])
	if(word[0].isupper):
		features[0] = 1
	features[1] = word[0:2]	
	features[2] = word[0:3]
	features[3] = word[0:4]
	features[4] = word[-2:]
	features[5] = word[-3:]
	features[6] = word[-4:]
	if (p == '-' for p in word):
		features[7] = 1
	if (word.isdigit()):
		features[8] = 1
	if (any(p.isdigit() for p in word)):
		features[9] = 1
	return features

if __name__ == '__main__':
	df_train, df_test, corpus, labels = readDataset('../data/train.txt')
	train_features = df_train['word'].apply(extractFeatures)
	print train_features 
