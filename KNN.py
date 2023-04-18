import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd 

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.datasets import fetch_20newsgroups
from imblearn.over_sampling import SMOTE 
from sklearn.linear_model import LogisticRegression
from imblearn.under_sampling import TomekLinks
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

import getopt
import sys
import numpy as np
import pandas as pd
import sklearn as sk
import imblearn
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import os.path
import pickle
from os import path
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score



df = pd.read_csv("datosProcesados.csv")
#print(df.head())

# Sacamos X y los labels
X= df['text']
y= df['__target__']
#smote = SMOTE()
#X,y = smote.fit_resample(X,y)
#(PEORES RESULTADOS)tomeklinks elimina los ejemplos cercanos entre las clases minoritarias y mayoritarias, con el objetivo de aumentar la separación entre las clases
#tomek_links = TomekLinks()
#(HORRIBLES RESULTADOS) Este método elimina aleatoriamente ejemplos de la clase mayoritaria para equilibrar el conjunto de datos.
#undersampler = RandomUnderSampler(random_state=42)
oversampler = RandomOverSampler(random_state=42)

# Vectorizamos en tf_idf todo
tfidf_vectorizer = TfidfVectorizer() 
tfidf_X = tfidf_vectorizer.fit_transform(X) # Se aplica tf idf a todos los datos

# Utilizamos SMOTE para balancear los datos. De esta forma hemos conseguido un 0.2 mas en accuracy
#X,y = smote.fit_resample(tfidf_X,y)
#X, y = tomek_links.fit_resample(tfidf_X, y)
#X, y = undersampler.fit_resample(tfidf_X, y)
X, y = oversampler.fit_resample(tfidf_X, y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 26)


# Ejecutamos Logistic Regression
# Utilizando el parametro C=10.0 se consigue una mayor regulación y es mejor para datasets pequeños porque evita el overfitting en ellos
# logistic_regression = LogisticRegression(C=10.0, random_state=42)
# logistic_regression.fit(X_train, y_train)

# # Conseguimos las predicciones
# y_pred = logistic_regression.predict(X_test)


W=['uniform','distance']
mResultado={'k':0,'d':0,'w':'','f-score':0}
for k in range(int(1),int(6)):
    if(k%2!=0):
        for d in range(int(1),int(3)):
            for w in W:
                #aplica el algoritmo KNN
                print("la k es: " + str(k))
                print("la d es: " + str(d))
                print('la w es:' + w)
                clf = KNeighborsClassifier(n_neighbors=int(k),
                                    weights=w,
                                    algorithm='auto',
                                    leaf_size=30,
                                    p=int(d))

                #Balancea el resultado se asignará un peso mayor a las clases menos representadas en el conjunto de datos.
                clf.class_weight = "balanced"

                # Explica lo que se hace en este paso
                #el clasificador se ajusta (fit) a los datos de entrenamiento (trainX, trainY), 
                # lo que significa que se ajustará a los patrones en los datos y aprenderá a clasificar nuevos datos.
                clf.fit(X_train, y_train)


                # Build up our result dataset

                # The model is now trained, we can apply it to our test set:

                predictions = clf.predict(X_test)
                probas = clf.predict_proba(X_test)


                print(f1_score(y_test, predictions, average=None))
                print(classification_report(y_test,predictions))
                print(confusion_matrix(y_test, predictions, labels=[1,0]))