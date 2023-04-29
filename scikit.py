from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
import sys
import pandas as pd
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt

dataset_name = sys.argv[1]
df = pd.read_csv(dataset_name)
X = df['text']

# aplicamos tf-idf en la columna text
vectorizer = CountVectorizer()
tf_matrix = vectorizer.fit_transform(X)

# creamos un elemento LatentDirichAtLocation con el número de tópicos = 10
lda_model = LatentDirichletAllocation(n_components=10)
lda_model.fit(tf_matrix)

resultados = []
topicos = {}
indice = 0
for j in tf_matrix:
    doc_lda = lda_model.transform(j)
    index = doc_lda.argmax()
    resultados.append(index)
    topic_words = []
    for i, topic in enumerate(lda_model.components_):
        top_words_idx = np.argsort(topic)[::-1][:10] # obtenemos las 10 palabras más relevantes
        topic_words.append([vectorizer.get_feature_names_out()[idx] for idx in top_words_idx])
    if index not in topicos:
        topicos[index] = topic_words
    indice = indice + 1

    if indice == 20:
        break

i = 0
for documento in X:
    print('El documento: ' + str(documento))
    print('pertenece al tópico: ' + str(resultados[i]))
    i += 1
    if i == 20:
        break

for i in topicos.keys():
    wordcloud = WordCloud().generate(' '.join([str(word) for word in topicos[i]]))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title('Topico ' + str(i))
    plt.show()
