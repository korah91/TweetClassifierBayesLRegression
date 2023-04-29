from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
import sys
import pandas as pd
dataset_name = sys.argv[1]
df = pd.read_csv(dataset_name)
X= df['text']
#aplicamos tf-idf en la columna text
vectorizer = CountVectorizer()
tf_matrix = vectorizer.fit_transform(X)
#creamos un elemnto LatentDirichAtLocation con el número de tópicos = 5
lda_model = LatentDirichletAllocation(n_components=5)
lda_model.fit(tf_matrix)
for j in tf_matrix:
    doc_lda = lda_model.transform(j)
    print(doc_lda)

    topic_words = []
    for i, topic in enumerate(lda_model.components_):
        top_words_idx = np.argsort(topic)[::-1][:10] # obtenemos las 10 palabras más relevantes
        topic_words.append([vectorizer.get_feature_names()[idx] for idx in top_words_idx])
    # imprime las 10 palabras más relevantes para cada tópico
    print(topic_words)

