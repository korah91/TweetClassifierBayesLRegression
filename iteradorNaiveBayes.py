from sklearn.metrics import f1_score
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
from imblearn.under_sampling import TomekLinks
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

df = pd.read_csv("datosProcesados.csv")
#print(df.head())

# Sacamos X y los labels
X= df['text']
y= df['__target__']
smote = SMOTE()
#oversampler = RandomOverSampler(random_state=42)
#tomek_links = TomekLinks()
undersampler = RandomUnderSampler(random_state=42)
# Vectorizamos en tf_idf todo
tfidf_vectorizer = TfidfVectorizer() 
tfidf_X = tfidf_vectorizer.fit_transform(X) # Se aplica tf idf a todos los datos

# Utilizamos SMOTE para balancear los datos. De esta forma hemos conseguido un 0.2 mas en accuracy
X,y = smote.fit_resample(tfidf_X,y)
#X, y = oversampler.fit_resample(tfidf_X, y)
#X, y = tomek_links.fit_resample(tfidf_X, y)
#X, y = undersampler.fit_resample(tfidf_X, y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 26)



def naiveBayes(X_train, X_test, y_train, y_test, alpha, fit_prior):

    # Ejecutamos naive bayes
    naive_bayes = MultinomialNB(alpha=alpha, fit_prior=fit_prior)

    naive_bayes.fit(X_train, y_train)

    # Conseguimos las predicciones
    y_pred = naive_bayes.predict(X_test)

    # Conseguimos el f-Score weighted. Utilizamos weighted porque hay desbalance
    f1_weighted = f1_score(y_test, y_pred, average='weighted')

    # Utilizamos la weighted porque hay desbalance
    print("Weighted F1 score:", f1_weighted)
    return f1_weighted



# hiperparametros de naive bayes a iterar
# Creo un array que va de 0 a 5 con step=0.1
alpha = np.arange(0, 5, 0.1)
fit_prior = [True, False]

fScores = []
# Ejecuto las iteraciones
for a in alpha:
    for f in fit_prior:
        fScore = naiveBayes(X_train, X_test, y_train, y_test, a, f)
        # Guardamos los datos de la iteracion
        iteracion = {'alpha': a, 'fit_prior': f, 'fScoreWeighted': fScore}
        # La anadimos al registro de iteraciones
        fScores.append(iteracion)

#print(fScores)
# Saco la mejor iteracion
i = 0
maxFscore = {'iteracion': 0, 'fScoreWeighted': 0}
for iteracion in fScores:
    if iteracion['fScoreWeighted'] > maxFscore['fScoreWeighted']:
        maxFscore = {'iteracion': i, 'fScoreWeighted': iteracion['fScoreWeighted']}
    i+=1

mejorIteracion = maxFscore['iteracion']
print("La mejor iteración es ",mejorIteracion, ", con los hiperparametros alpha=",fScores[mejorIteracion]['alpha'], ", fit_prior=", fScores[mejorIteracion]['fit_prior'], " y con fScoreWeighted=",fScores[mejorIteracion]['fScoreWeighted'])
