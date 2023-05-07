from joblib import Parallel, delayed

import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

import matplotlib.pyplot as plt
import numpy as np
from gensim.models.ldamulticore import LdaMulticore
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





# Paso todos los textos a una lista
textos = df.text.values.tolist()

# Paso cada texto de cada tweet de una string a una lista de palabras
data_words = list(sent_to_words(textos))

# Se crea el diccionario de las palabras; cada palabra unica contiene un identificador. Sirve para crear el corpus
id2word = corpora.Dictionary(data_words)

# Se crea el corpus
corpus = [id2word.doc2bow(text) for text in data_words]
#print(corpus[0])
# Cada palabra: (word_id, word_frequency). Si es (47,3) quiere decir que la palabra con id 47 aparece 3 veces en el documento

modelos = []

def funcionLDA_model(corpus, id2word, num_topics, iterations):
    lda_model = LdaMulticore(corpus=corpus,  
                                            id2word=id2word,
                                            num_topics=num_topics, 
                                            random_state=100,
                                            chunksize=100,
                                            passes=20,
                                            iterations=iterations,
                                            eval_every= None,
                                            per_word_topics=True,
                                            workers=8)
    # Se anade el modelo a la lista para luego guardar el mejor
    modelos.append(lda_model)
    
    # La idea es tener la mejor similitud entre los textos de un topico mientras que se minimiza la similitud con los textos de otros
    # La coherencia relaciona la distancia intracluster con la distancia intercluster
    coherence_model_lda = CoherenceModel(model=lda_model, texts=data_words, dictionary=id2word, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print('Iteracion: ',i, ', Coherence Score: ', coherence_lda, ', n_topics: ', num_topics)
    
    return coherence_lda


# Imprime los topicos; por cada topico muestra su id y luego las palabras mas frecuentes con la frecuencia de esa palabra en ese topico
#print(lda_model.print_topics())
registroIteraciones = []

i = 1
# Itera
for n_topicos in np.arange(2, 5, 1):
    
    # Guardamos los datos de la iteracion
    coherence = funcionLDA_model(corpus, id2word, n_topicos, 10)
    iteracion = {'iteracion': i,'n_topicos': n_topicos, 'coherence': coherence}
    # La anadimos al registro de iteraciones
    registroIteraciones.append(iteracion)
    i+=1

# Saco la mejor iteracion
i = 1
maxCoherence = {'iteracion': 0, 'coherence': 0}
for iteracion in registroIteraciones:
    if iteracion['coherence'] > maxCoherence['coherence']:
        maxCoherence = {'iteracion': i, 'coherence': iteracion['coherence']}
    i+=1

mejorIteracion = maxCoherence['iteracion']
print("La mejor iteración es la ",registroIteraciones[mejorIteracion]['iteracion'], ". coherence=",registroIteraciones[mejorIteracion]['coherence'], ", n_topicos: ", registroIteraciones[mejorIteracion]['n_topicos'])


array_n_topicos = [i['n_topicos'] for i in registroIteraciones]
array_coherence = [i['coherence'] for i in registroIteraciones]


# Se guarda el mejor modelo
modelos[registroIteraciones[mejorIteracion]['iteracion'] - 1].save("modeloLDA")

plt.plot(array_n_topicos, array_coherence)
plt.xlabel("Number of Topics")
plt.ylabel("Coherence Score")
plt.show()
