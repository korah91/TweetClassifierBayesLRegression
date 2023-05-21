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
from sklearn.metrics import f1_score




df = pd.read_csv("datosProcesados.csv")
#print(df.head())

# Sacamos X y los labels
X = df['text']
y = df['__target__']

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


def logisticRegression(X_train, X_test, y_train, y_test, c):

    # Ejecutamos Logistic Regression
    # Utilizando el parametro C=10.0 se consigue una mayor regulación y es mejor para datasets pequeños porque evita el overfitting en ellos
    logistic_regression = LogisticRegression(C=c, random_state=42)
    logistic_regression.fit(X_train, y_train)

    # Conseguimos las predicciones
    y_pred = logistic_regression.predict(X_test)
    
    # Conseguimos el f-Score weighted. Utilizamos weighted porque hay desbalance
    f1_weighted = f1_score(y_test, y_pred, average='weighted')

    # Utilizamos la weighted porque hay desbalance
    print("Weighted F1 score:", f1_weighted)
    return f1_weighted
 

# Los parametros sobre los que iterar
C = np.arange(0, 100, 2.5)

fScores = []

for c in C:
    fScore = logisticRegression(X_train, X_test, y_train, y_test, c)
    # Guardamos los datos de la iteracion
    iteracion = {'c': c, 'fScoreWeighted': fScore}
    # La anadimos al registro de iteraciones
    fScores.append(iteracion)


# Saco la mejor iteracion
i = 0
maxFscore = {'iteracion': 0, 'fScoreWeighted': 0}
for iteracion in fScores:
    if iteracion['fScoreWeighted'] > maxFscore['fScoreWeighted']:
        maxFscore = {'iteracion': i, 'fScoreWeighted': iteracion['fScoreWeighted']}
    i+=1

mejorIteracion = maxFscore['iteracion']
print("La mejor iteración es ",mejorIteracion, ", con el hiperparametro c=",fScores[mejorIteracion]['c'], " y con fScoreWeighted=",fScores[mejorIteracion]['fScoreWeighted'])
