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
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.datasets import fetch_20newsgroups
from imblearn.over_sampling import SMOTE 


df = pd.read_csv("datosProcesados.csv")
#print(df.head())

# Sacamos X y los labels
X= df['text']
y= df['__target__']
smote = SMOTE()
#X,y = smote.fit_resample(X,y)


# Vectorizamos en tf_idf todo
tfidf_vectorizer = TfidfVectorizer() 
tfidf_X = tfidf_vectorizer.fit_transform(X) # Se aplica tf idf a todos los datos

# Utilizamos SMOTE para balancear los datos. De esta forma hemos conseguido un 0.2 mas en accuracy
X,y = smote.fit_resample(tfidf_X,y)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 26)


# Ejecutamos naive bayes
naive_bayes_classifier = MultinomialNB()

naive_bayes_classifier.fit(X_train, y_train)

# Conseguimos las predicciones
y_pred = naive_bayes_classifier.predict(X_test)

print(accuracy_score(y_pred, y_test))





#################### Representar la matriz de confusion

#print(classification_report(y_test,y_pred))

cnf_matrix = confusion_matrix(y_test,y_pred)

print(cnf_matrix)
clases = ['Negative', 'Neutral', 'Positive']
sns.heatmap(cnf_matrix, annot=True, cmap='Blues', fmt='g', xticklabels=clases, yticklabels=clases)
plt.title("Matriz de confusion")
plt.xlabel("Real")
plt.ylabel("Predicción")
plt.savefig('mi_matriz_de_confusion.jpg')
plt.show()
print(classification_report(y_test, y_pred))


y_train.value_counts().plot(kind='bar')