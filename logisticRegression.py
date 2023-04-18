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
logistic_regression = LogisticRegression(C=10.0, random_state=42)
logistic_regression.fit(X_train, y_train)

# Conseguimos las predicciones
y_pred = logistic_regression.predict(X_test)

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