import numpy as np
import pandas as pd

def readData(filepath):
	trainFile = open(filepath+'train.txt', 'r')
	testFile = open(filepath+'test.txt','r')
	df_train = pd.read_table(trainFile, delim_whitespace=True, names=('word','posTag','chunkTag'))
	trainFile.close()

	df_test = pd.read_table(testFile, delim_whitespace=True, names=('word','posTag','chunkTag'))
	testFile.close()

	corpus = df_train['word'].unique()
	tags = df_train['posTag'].unique()	

	#print corpus.shape (19122 vocabulary size, including puctuations)
	#print tags.shape   (44 pos tags, including punctuations)

	return df_train, df_test, corpus, tags


