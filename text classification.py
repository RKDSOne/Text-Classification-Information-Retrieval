from numpy import genfromtxt
import numpy as np
from collections import OrderedDict
import csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.decomposition import TruncatedSVD
import nltk
from  nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
import string
from sklearn.svm import LinearSVC


target_train_vector=[];
target_train_list=[];
with open('train.csv', 'rb') as f:
    reader = csv.reader(f)
    next(reader, None)
    for row in reader:
    	target_train_vector.append(row[0]);
    	target_train_list.append(', '.join(row[1:]));




stopwords = stopwords.words('english')



vectorizer = TfidfVectorizer(ngram_range=(1, 2),stop_words=stopwords)
X_train = vectorizer.fit_transform(target_train_list)
# PCA reducted for sparse matrices
#pca = TruncatedSVD(n_components=5000)
#X_reduced_train = pca.fit_transform(X_train)
Y_train = target_train_vector;
#print sum(pca.explained_variance_)


target_test_list=[];
with open('test.csv', 'rb') as f:
    reader = csv.reader(f)
    next(reader, None)
    for row in reader:
    	target_test_list.append(', '.join(row[1:]));



X_test = vectorizer.transform(target_test_list);
#X_reduced_test = pca.transform(X_test)

svc_classifier = LinearSVC().fit(X_train, Y_train)
final_predictions = svc_classifier.predict(X_test)


#print final_predictions;
##https://www.kaggle.com/wiki/GettingStartedWithPythonForDataScience
predicted_values = [[str(ind+1),int(x)] for ind, x in enumerate(final_predictions)]

#from numpy import genfromtxt, savetxt
np.savetxt('Submission_SI650_HW3_v22.csv', predicted_values, delimiter=',',header='Id,Category', fmt="%s", comments = '')



