import numpy as np
import csv
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

target_train_vector = []
target_train_list = []
with open('train.csv', 'rb') as f:
    reader = csv.reader(f)
    next(reader, None)
    for row in reader:
        target_train_vector.append(row[0])
        target_train_list.append(', '.join(row[1:]))

stopwords = stopwords.words('english')

vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words=stopwords)
X_train = vectorizer.fit_transform(target_train_list)
Y_train = target_train_vector

target_test_list = []
with open('test.csv', 'rb') as f:
    reader = csv.reader(f)
    next(reader, None)
    for row in reader:
        target_test_list.append(', '.join(row[1:]))

X_test = vectorizer.transform(target_test_list)

svc_classifier = LinearSVC().fit(X_train, Y_train)
final_predictions = svc_classifier.predict(X_test)

# https://www.kaggle.com/wiki/GettingStartedWithPythonForDataScience
predicted_values = [[str(ind + 1), int(x)] for ind, x in enumerate(final_predictions)]

np.savetxt('Submission.csv', predicted_values, delimiter=',', header='Id,Category', fmt="%s", comments='')
