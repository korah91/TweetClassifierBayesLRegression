import os
import csv
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

# Nombre del archivo del dataset

dataset_name = sys.argv[1]

# Leer el dataset
df = pd.read_csv(dataset_name)

# Seleccionar los documentos negativos
X = df[df['__target__'] == 0]['text']  # Cambiar 0 por el valor correspondiente

# Aplicar TF en la columna "text"
vectorizer = CountVectorizer()
tf_matrix = vectorizer.fit_transform(X)

# Crear un modelo LDA con 20 tópicos
lda_model = LatentDirichletAllocation(n_components=20)
lda_model.fit(tf_matrix)

# Obtener las distribuciones de tópicos para todos los documentos
doc_lda = lda_model.transform(tf_matrix)

# Obtener los tópicos para cada documento
resultados = np.argmax(doc_lda, axis=1)

# Imprimir las palabras más relevantes de cada tópico
for i, topic in enumerate(lda_model.components_):
    print("Topico %d:" % (i))
    print(" ".join([vectorizer.get_feature_names_out()[j] for j in topic.argsort()[:-10 - 1:-1]]))

# Crear un diccionario con las palabras más relevantes de cada tópico
topicos = {}
for i, topic in enumerate(lda_model.components_):
    top_words_idx = np.argsort(topic)[::-1][:10]
    topic_words = [vectorizer.get_feature_names_out()[idx] for idx in top_words_idx]
    topicos[i] = topic_words

# Crear la carpeta si no existe
if not os.path.exists("TopicosNegativos"):
    os.makedirs("TopicosNegativos")

# Crear una WordCloud para cada tópico y guardarla en un archivo
for i in topicos.keys():
    wordcloud = WordCloud().generate(" ".join([str(word) for word in topicos[i]]))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"Topico {i}")
    plt.savefig(f"TopicosNegativos/mi_wordcloud_del_topico_{i}.jpg")

# Guardar los resultados en un archivo CSV
with open("resultados.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["documento", "topico"])
    for i, doc in enumerate(X):
        writer.writerow([doc, resultados[i]])
