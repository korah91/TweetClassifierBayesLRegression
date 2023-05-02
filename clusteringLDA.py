# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt

import pandas as pd 


# Tokeniza cada texto en una lista de palabras
def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations


#https://radimrehurek.com/gensim/auto_examples/tutorials/run_lda.html
# Recomienda hacer una tabla con parámetros con las pruebas que queremos hacer:
# Primero se hace una tabla con los alfas y betas


#dataset_name = sys.argv[1]
#df = pd.read_csv(dataset_name)
df = pd.read_csv("datosProcesados.csv")



# Seleccionar las filas en las que la opinion sea negativa, mas tarde podemos probar con las positivas y las neutrales
X = df.loc[df['__target__']] == 'negative'

# Paso todos los textos a una lista
textos = df.text.values.tolist()

# Paso cada texto de cada tweet de una string a una lista de palabras
data_words = list(sent_to_words(textos))

# Se crea el diccionario de las palabras; cada palabra unica contiene un identificador. Sirve para crear el corpus
id2word = corpora.Dictionary(data_words)

# Se crea el corpus
corpus = [id2word.doc2bow(text) for text in data_words]
print(corpus[0])
# Cada palabra: (word_id, word_frequency). Si es (47,3) quiere decir que la palabra con id 47 aparece 3 veces en el documento


lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,  
                                           id2word=id2word,
                                           num_topics=20, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=20,
                                           alpha='auto',
                                           eta='auto',
                                           iterations=100,
                                           eval_every= None,
                                           per_word_topics=True)

# Imprime los topicos; por cada topico muestra su id y luego las palabras mas frecuentes con la frecuencia de esa palabra en ese topico
print(lda_model.print_topics())

# La coherencia relaciona la distancia intracluster con la distancia intercluster
coherence_model_lda = CoherenceModel(model=lda_model, texts=data_words, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)