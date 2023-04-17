# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # for data visualization purposes
import seaborn as sns # for statistical data visualization
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline



df = pd.read_csv("datosProcesados.csv")
print(df.head())

X= df['text']
y= df['__target__']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 26)



#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@https://www.kaggle.com/code/ankumagawa/movie-sentiment-analysis-with-tf-idf-naive-bayes?scriptVersionId=99900601&cellId=20




tfidf_vectorizer = TfidfVectorizer() 

tfidf_train_vectors = tfidf_vectorizer.fit_transform(X_train) #applying tf idf to training data

tfidf_test_vectors = tfidf_vectorizer.transform(X_test) #applying tf idf to training data

print(tfidf_train_vectors)
# check the dimension of the data now: 
print("n_samples: %d, n_features: %d" % tfidf_train_vectors.shape)


#naive bayes classifier
naive_bayes_classifier = MultinomialNB()

naive_bayes_classifier.fit(tfidf_train_vectors, y_train)

#predicted y
y_pred = naive_bayes_classifier.predict(tfidf_test_vectors)







###

#print(classification_report(y_test,y_pred))

cnf_matrix = confusion_matrix(y_test,y_pred)

print(cnf_matrix)

sns.heatmap(cnf_matrix, annot=True, cmap='Blues', fmt='g')
plt.savefig('mi_matriz_de_confusion.jpg')
#plt.show()
print(classification_report(y_test, y_pred))


y_train.value_counts().plot(kind='bar')