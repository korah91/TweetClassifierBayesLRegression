import csv
import os
import getopt
import sys
import numpy as np
import pandas as pd
import sklearn as sk
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from geotext import GeoText


#API KEY PARA CONSEGUIR COORDENADAS POR LA CIUDAD

#API DE FRAN
#api_key = "46b90a6f42b1415da5b4ec372a1a4b2e"

#API DE ANDREEA
#api_key = "05baa7b99edb489abd29b3ba70e7adbc"

#API DE JOEL
#api_key = ""

#API DE MARIO
#api_key = ""

from opencage.geocoder import OpenCageGeocode

# Instalar openCage: pip3 install opencage
#geocoder = OpenCageGeocode(api_key)

# AHORA UTILIZAMOS pip install geopy PARA NO UTILIZAR LA API, QUE TIENE POCOS USOS
from geopy.geocoders import Nominatim
geolocator = Nominatim(user_agent="proyectoSADGE")

#  Se pasan todos los atributos de texto a unicode
def coerce_to_unicode(x):
    if sys.version_info < (3, 0):
        if isinstance(x, str):
            return unicode(x, 'utf-8')
        else:
            return unicode(x)
    else:
        return str(x)
    
    
# Se utiliza esta funcion para dos columnas; para tweet_location y para tweet_coord
def conseguir_ciudad(row):
    # Accedo a la columna
    user_timezone = row['user_timezone']
    
    if user_timezone == 'Eastern Time (US & Canada)':
        user_timezone = 'New York City, New York'
    elif user_timezone == 'Central Time (US & Canada)':
        user_timezone = 'Austin, Texas'
    elif user_timezone == 'Mountain Time (US & Canada)':
        user_timezone = 'Denver, Colorado'
    elif user_timezone == 'Pacific Time (US & Canada)':
        user_timezone = 'San Francisco, California'
    elif user_timezone == 'Atlantic Time (Canada)':
        user_timezone = 'Nova Scotia, Canada'   
    # Si no se puede utilizar 
    elif user_timezone == 'nan':
        # Obtenemos la aerolinea
        aerolinea = row['airline']
        if aerolinea == 'United':
            user_timezone = 'Chicago, Illinois'
        elif aerolinea == 'Southwest':
            user_timezone = 'Dallas, Texas'
        elif aerolinea == 'Delta':
            user_timezone = 'Atlanta, Georgia'
        elif aerolinea == 'US airways':
            user_timezone = 'Alexandria, Virginia'
        elif aerolinea == 'Virgin America':
            user_timezone = 'Burlingame, California'
    #El resto de ciudades 
    else:
        user_timezone = row['user_timezone']

    return user_timezone

mapaCiudades={}


def conseguir_coordenadas(ciudad):
    #results = geocoder.geocode(query)
    #coordenadas = [results[0]['geometry']['lat'], results[0]['geometry']['lng']]
    #Realizamos una excepcion para Nonetype
    try:
        if(ciudad not in mapaCiudades):
            location = geolocator.geocode(ciudad, timeout=None)   
            #location = geocoder.geocode(timezone, timeout=None)
            mapaCiudades[str(ciudad)] = [location.latitude, location.longitude]
            # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ ESTO CON LA API CREO QUE NO HARIA FALTA, POIRQUE ES MUY COCHINO Y CON LA API IGUAL VA MEJOR @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        return mapaCiudades.get(ciudad)

    except:
        return[0,0]
    
def tratar_tweet_location(dataset):
    # Convertir a minuscula
    dataset['tweet_location'] = dataset['tweet_location'].str.lower()

    # Eliminar caracteres especiales y signos de puntuacion
    dataset['tweet_location'] = dataset['tweet_location'].str.replace('[^\w\s]', '')
    
    # Recorremos cada fila reconociendo la ciudad
    for index, row in dataset.iterrows():
        row['tweet_location'] = str(row['tweet_location'])
        if row['tweet_location'] == 'nan':
            row['tweet_location'] = conseguir_ciudad(row)
        ciudad = GeoText(row['tweet_location']).cities
        
        # Si el tweet_location tiene una ciudad se reemplaza el valor limpio
        if (len(ciudad) != 0):
            dataset.loc[index, 'tweet_location'] = ciudad[0]
        else:
            # Si tiene un missing value o un dato incorrecto se intuye la ciudad utilizando el huso horario o la sede de la aerolinea
            dataset.loc[index, 'tweet_location'] = str(conseguir_ciudad(row))
    return dataset


# @@@@@@@@@@@@@@@@@@@@@ Tratamos la columna tweet_coord @@@@@@@@@@@@@@@@@@@@@
def tratar_tweet_coord(dataset):
    # Se utiliza la API para conseguir las coordenadas
    # Por cada fila: index es el numero de fila. row es la fila
    for index, row in dataset.iterrows():
        # Si tweet_coord es missingValue
        
        #print(row['tweet_coord'], ", tipo: ", type(row['tweet_coord']))
        row['tweet_coord'] = str(row['tweet_coord'])
        if row['tweet_coord']=='nan':

            # Segun el timezone se asigna una ciudad
            ciudad = conseguir_ciudad(row)    
            
            #actualizar mapa de coordenadas si es nueva ciudad
            conseguir_coordenadas(ciudad)              


            # Conseguimos las coordenadas
            print("index: ",index, ",tweet_coord: ", dataset.loc[index, 'tweet_coord'], " --> ", mapaCiudades[str(ciudad)])

            dataset.loc[index, 'tweet_coord'] = str(conseguir_coordenadas(ciudad))

    return dataset

def tratar_text(dataset):
    #print(dataset.head(5))
    # Convertir a minuscula
    dataset['text'] = dataset['text'].str.lower()

    # Eliminar caracteres especiales y signos de puntuacion
    dataset['text'] = dataset['text'].str.replace('[^\w\s]', '')

    # Quitar stopwords
    stop_words = set(stopwords.words('english'))
    dataset['text'] = dataset['text'].apply(lambda x: ' '.join([word for word in word_tokenize(x) if word.lower() not in stop_words]))

    # Lematizar
    stemmer = PorterStemmer()
    dataset['text'] = dataset['text'].apply(lambda x: ' '.join([stemmer.stem(word) for word in word_tokenize(x)]))

    return dataset


def borrar_features(dataset,features):
    #borra cada feature del dataset
    for feature in features: 
        dataset= dataset.drop(feature, axis=1)
    return dataset

#Abrir el fichero .csv y cargarlo en un dataframe de pandas
ml_dataset = pd.read_csv("TweetsTrainDev.csv")


#comprobar que los datos se han cargado bien. Cuidado con las cabeceras, la primera línea por defecto la considerara como la que almacena los nombres de los atributos
# comprobar los parametros por defecto del pd.read_csv en lo referente a las cabeceras si no se quiere lo comentado

#print(ml_dataset.head(5))

# Se introducen los nombres de las columnas
ml_dataset = ml_dataset[
    ['tweet_id','airline_sentiment', 'airline_sentiment_confidence','negativereason','negativereason_confidence','airline','name','retweet_count','text','tweet_coord','tweet_created','tweet_location','user_timezone']
    ]

# Se guardan los nombres de las columnas de valores categóricos
categorical_features = ['airline_sentiment', 'negativereason','airline', 'name', 'text', 'tweet_location', 'user_timezone']
# Se guardan los nombres de las columnas de valores numericos
numerical_features = ['tweet_id','airline_sentiment_confidence','negativereason_confidence','retweet_count']
text_features = []
for feature in categorical_features:
    ml_dataset[feature] = ml_dataset[feature].apply(coerce_to_unicode)

for feature in text_features:
    ml_dataset[feature] = ml_dataset[feature].apply(coerce_to_unicode)

for feature in numerical_features:
    if ml_dataset[feature].dtype == np.dtype('M8[ns]') or (
            hasattr(ml_dataset[feature].dtype, 'base') and ml_dataset[feature].dtype.base == np.dtype('M8[ns]')):
        ml_dataset[feature] = datetime_to_epoch(ml_dataset[feature])
    else:
        ml_dataset[feature] = ml_dataset[feature].astype('double')



ml_dataset = tratar_tweet_location(ml_dataset)
ml_dataset = tratar_text(ml_dataset)
ml_dataset = tratar_tweet_coord(ml_dataset)
    



    
# Los valores posibles de la clase a predecir Especie. 
# Puede ser de 3 clases
target_map = {'negative': 0, 'neutral': 1, 'positive': 2}

# Columna que se utilizará
ml_dataset['__target__'] = ml_dataset['airline_sentiment'].map(str).map(target_map)
del ml_dataset['airline_sentiment']

# Se borran las filas en las que la clase a predecir no aparezca
ml_dataset = ml_dataset[~ml_dataset['__target__'].isnull()]

#print(ml_dataset.head(5))

# Lista con los atributos que cuando faltan en una instancia hagan que se tenga que borrar
drop_rows_when_missing = []

# Lista con los atributos que cuando faltan en una instancia se tenga que corregir haciendo la media, mediana, etc. del resto
impute_when_missing = [{'feature': 'negativereason_confidence', 'impute_with': 'MEAN'},
                        {'feature': 'negativereason', 'impute_with': 'MODE'},
                        ]
                        
# Se borran las filas en las que falten los atributos que haya en la lista drop_rows_when_missing
for feature in drop_rows_when_missing:
    ml_dataset = ml_dataset[ml_dataset[feature].notnull()]
    
    print('Dropped missing records in %s' % feature)

# Se corrigen todos los datos faltantes de los atributos en la lista impute_when_missing dependiendo de como se deban tratar
# En este caso todos se corrigen con la media del resto de instancias
for feature in impute_when_missing:
    if feature['impute_with'] == 'MEAN':
        v = ml_dataset[feature['feature']].mean()
    elif feature['impute_with'] == 'MEDIAN':
        v = ml_dataset[feature['feature']].median()
    elif feature['impute_with'] == 'CREATE_CATEGORY':
        v = 'NULL_CATEGORY'
    elif feature['impute_with'] == 'MODE':
        print("SE ARREGLA NEGATIVEREASON")
        valores_de_moda = ml_dataset[feature['feature']].value_counts()
        print(valores_de_moda)
        # Si en el dataset la moda es nan, no hay que poner nan en el procesado
        if valores_de_moda[0] == 'nan':
            v = ml_dataset[feature['feature']].value_counts().index[1]
        else:
            v = ml_dataset[feature['feature']].value_counts().index[0]
        
    elif feature['impute_with'] == 'CONSTANT':
        v = feature['value']
      
    ml_dataset[feature['feature']] = ml_dataset[feature['feature']].fillna(v)
    print('Imputed missing values in feature %s with value %s' % (feature['feature'], coerce_to_unicode(v)))


ml_dataset.to_csv("datosProcesados.csv", sep=',', encoding='utf-8', index=True, header=True)

print(ml_dataset.head(5))
print("Se ha llamado a la API ", len(mapaCiudades), " veces.")

print(type(ml_dataset.loc[0, 'negativereason']))

