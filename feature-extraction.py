import sklearn
import os
import numpy as np
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.utils import shuffle
from sklearn.metrics import classification_report
from nltk.tokenize import word_tokenize
from scipy.sparse import csr_matrix
from collections import Counter
from sklearn.model_selection import GridSearchCV
import spacy

nlp = spacy.load('en_vectors_web_lg')


def extract_features(data):
	res = []
	for entry in data:
		vector = []
		toked_entry = entry.split()
		indexes = [i for i, item in enumerate(token_entry) if item.startswith('<head>')]
		for i in indexes: # this should only have 1, the location of head
			# nlp turns each of the words into a vector
			print(token_entry[i-2:i] + token_entry[i+1:i+2])
			vector.append([nlp(x)[0].vector for x in (token_entry[i-2:i] + token_entry[i+1:i+2])])
			vector.append(vector[0] - vector[1]) # word-2 - word-1
			vector.append(vector[1] - vector[2]) # word-1 - word+1
			vector.append(vector[2] - vector[3]) # word+1 = word+2
		res.append(vector)

	# This changes the list of lists into a more compact array
	# representation that only stores non-zero values
	#res = csr_matrix(res)
	return res


# This asks the user for the datafile that they want to extract features from.
def getInputFile():
	bad = True
	while bad:
		try:
			fileName = input("Enter file name: ")
			# Open file for input
			f = open(fileName, "r")
			bad = False
		except Exception as err:
			print("Please enter a valid file name:")
	return f.read()



def main():
	# Read in dataset
	print("Reading in dataset...")
	train_text_data, train_Y = read_dataset("aclImdb/train")
	test_text_data, test_Y = read_dataset("aclImdb/test")
	print(Counter(test_Y))
	# Now we need to extract features from the text data
	print("Extracting features...")
	train_X = extract_features(train_text_data)
	test_X = extract_features(test_text_data)
	#params = {'n_estimators': [10, 20, 30, 100],
	  #        'criterion': ['gini', 'entropy']}
	svm_model = SVC(gamma='auto')
	model = GridSearchCV(svm_model)
	print("Training...")
	model.fit(train_X, train_Y)
	preds = model.predict(test_X)
	print(classification_report(test_Y, preds, digits=6))


if __name__ == '__main__':
	main()

