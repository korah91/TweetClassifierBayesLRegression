import pandas as pd
from gensim.models.ldamodel import LdaModel
from gensim import corpora
from gensim.models.tfidfmodel import TfidfModel
import sys

# Cargar el modelo LDA entrenado previamente
lda_model = LdaModel.load('modeloLDA')

# Cargar el conjunto de datos de prueba
test_data = pd.read_csv(sys.argv[1])

# Preprocesar los documentos de prueba
docs = [doc.split() for doc in test_data['texto'].tolist()]

# Crear un diccionario de términos a partir de los documentos
dictionary = corpora.Dictionary(docs)

# Convertir los documentos a una matriz de términos de documentos (DTM)
dtm = [dictionary.doc2bow(doc) for doc in docs]

# Convertir la matriz DTM en una matriz TFIDF
tfidf = TfidfModel(dtm)

# Convertir la matriz TFIDF a un formato que pueda ser usado por el modelo LDA
corpus = tfidf[dtm]

# Obtener las predicciones de los tópicos para cada documento en el corpus
predictions = [lda_model.get_document_topics(doc) for doc in corpus]

# Hacer algo con las predicciones obtenidas, por ejemplo:
for doc_pred in predictions:
    max_topic = max(doc_pred, key=lambda x: x[1])[0]
    print(f'La feature predicha para este documento es: {max_topic}')
