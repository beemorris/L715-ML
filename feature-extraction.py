import sklearn
import os
import numpy as np
from sklearn import svm
from sklearn.svm import svc
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
	"""
	data is a list of strings here where each string represents a document
	that we're trying to determine the sentiment of.

	This is a very simple feature extraction method that just takes a
	list of positive words, a list of negative words and then gets the count
	of both of these word groups in the given string. An additional feature is
	used to signal if negation was present.
	"""
	res = []
	for entry in data:
		vector = []
		token_entry = entry.split()
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
def get_input_file():
	f = None
	tf = None
	bad = True
	while bad:
		try:
			file_name = input("Enter data file name: ")
			# Open file for input
			f = open(file_name, "r").read()
			bad = False
		except Exception as err:
			print("Please enter a valid file name:")
	bad = True
	while bad:
		try:
			file_name = input("Enter data file name: ")
			# Open file for input
			tf = open(file_name, "r").readlines()
			bad = False
		except Exception as err:
			print("Please enter a valid file name:")
	return f, tf



def main():
	# Read in dataset
	print("Reading in dataset...")
	train_text_data, train_Y = get_input_file()
	test_text_data, test_Y = get_input_file()
	print(Counter(test_Y))
	# Now we need to extract features from the text data
	print("Extracting features...")
	train_X = extract_features(train_text_data)
	test_X = extract_features(test_text_data)
	# params = {'n_estimators': [10, 20, 30, 100],
	#        'criterion': ['gini', 'entropy']}
	svm_model = SVC(gamma='auto')
	model = GridSearchCV(svm_model)
	print("Training...")
	model.fit(train_X, train_Y)
	preds = model.predict(test_X)
	print(classification_report(test_Y, preds, digits=6))


if __name__ == '__main__':
	main()
