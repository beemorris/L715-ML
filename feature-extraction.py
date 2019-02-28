import sklearn
import os
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from nltk.tokenize import word_tokenize
from collections import Counter
from sklearn.model_selection import GridSearchCV
import spacy

nlp = spacy.load('en_vectors_web_lg')


def extract_features(data):
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
	# the [1:] is to exclude the first couple lines after splitting on <instance
	train_X = extract_features(train_text_data.split('<instance')[1:])
	test_X = extract_features(test_text_data)
	# params = {'n_estimators': [10, 20, 30, 100],}
	svm_model = SVC(gamma='auto')
	# model = GridSearchCV(svm_model)
	print("Training...")
	svm_model.fit(train_X, train_Y)
	preds = svm_model.predict(test_X)
	print(classification_report(test_Y, preds, digits=6))


if __name__ == '__main__':
	main()
