

# 
compania = 

#  Se pasan todos los atributos de texto a unicode
def coerce_to_unicode(x):
    if sys.version_info < (3, 0):
        if isinstance(x, str):
            return unicode(x, 'utf-8')
        else:
            return unicode(x)
    else:
        return str(x)

#Abrir el fichero .csv y cargarlo en un dataframe de pandas
ml_dataset = pd.read_csv(iFile)

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
numerical_features = ['tweet_id','airline_sentimen_confidence','negativereason_confidence','retweet_count']
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


# Los valores posibles de la clase a predecir Especie. 
# Puede ser de 3 clases
target_map = {'negative': 0, 'neutral': 1, 'positive': 2}

# Columna que se utilizará
ml_dataset['__target__'] = ml_dataset['airline_sentiment'].map(str).map(target_map)
del ml_dataset['airline_sentiment']

# Se borran las filas en las que la clase a predecir no aparezca
ml_dataset = ml_dataset[~ml_dataset['__target__'].isnull()]
print(f)
print(ml_dataset.head(5))

# Se crean las particiones de Train/Test
train, test = train_test_split(ml_dataset,test_size=0.2,random_state=42,stratify=ml_dataset[['__target__']])
print(train.head(5))
print(train['__target__'].value_counts())
print(test['__target__'].value_counts())

# Lista con los atributos que cuando faltan en una instancia hagan que se tenga que borrar
drop_rows_when_missing = []

# Lista con los atributos que cuando faltan en una instancia se tenga que corregir haciendo la media, mediana, etc. del resto
impute_when_missing = [{'feature': 'negativereason_confidence', 'impute_with': 'MEAN'},
                        {'feature': 'negativereason', 'impute_with': 'MODE'},
                        {'feature': 'Largo de petalo', 'impute_with': 'MEAN'},
                        {'feature': 'Ancho de petalo', 'impute_with': 'MEAN'}]
                        
# Se borran las filas en las que falten los atributos que haya en la lista drop_rows_when_missing
for feature in drop_rows_when_missing:
    train = train[train[feature].notnull()]
    test = test[test[feature].notnull()]
    print('Dropped missing records in %s' % feature)

# Se corrigen todos los datos faltantes de los atributos en la lista impute_when_missing dependiendo de como se deban tratar
# En este caso todos se corrigen con la media del resto de instancias
for feature in impute_when_missing:
    if feature['impute_with'] == 'MEAN':
        v = train[feature['feature']].mean()
    elif feature['impute_with'] == 'MEDIAN':
        v = train[feature['feature']].median()
    elif feature['impute_with'] == 'CREATE_CATEGORY':
        v = 'NULL_CATEGORY'
    elif feature['impute_with'] == 'MODE':
        v = train[feature['feature']].value_counts().index[0]
    elif feature['impute_with'] == 'CONSTANT':
        v = feature['value']
    train[feature['feature']] = train[feature['feature']].fillna(v)
    test[feature['feature']] = test[feature['feature']].fillna(v)
    print('Imputed missing values in feature %s with value %s' % (feature['feature'], coerce_to_unicode(v)))


# Lista con los métodos para escalar cada atributo numerico 
rescale_features = {'Largo de sepalo': 'AVGSTD', 
                    'Ancho de sepalo': 'AVGSTD', 
                    'Largo de petalo': 'AVGSTD',
                    'Ancho de petalo': 'AVGSTD'}

# Se reescala
for (feature_name, rescale_method) in rescale_features.items():
    if rescale_method == 'MINMAX':
        _min = train[feature_name].min()
        _max = train[feature_name].max()
        scale = _max - _min
        shift = _min
    else:
        shift = train[feature_name].mean()
        scale = train[feature_name].std()
    if scale == 0.:
        del train[feature_name]
        del test[feature_name]
        print('Feature %s was dropped because it has no variance' % feature_name)
    else:
        print('Rescaled %s' % feature_name)
        train[feature_name] = (train[feature_name] - shift).astype(np.float64) / scale
        test[feature_name] = (test[feature_name] - shift).astype(np.float64) / scale




# Valores del conjunto Train
trainX = train.drop('__target__', axis=1)
#trainY = train['__target__']

# Valores del conjunto Test
testX = test.drop('__target__', axis=1)
#testY = test['__target__']

# Etiquetas del conjunto Train
trainY = np.array(train['__target__'])
# Etiquetas del conjunto Test
testY = np.array(test['__target__'])

# Explica lo que se hace en este paso
# Se realiza undersampling con la funcion de la libreria imbalanced-learn.
# El undersampling consiste en borrar instancias de la clase dominante para equilibrar el dataset

# Utilizamos un dict como sampling strategy
sampling_strategy = {0: 10, 1: 10, 2: 10}
undersample = RandomUnderSampler(sampling_strategy=sampling_strategy)

# Se reemplazan los conjuntos Train/Test con unos conjuntos a los que se les ha realizado undersampling
trainXUnder,trainYUnder = undersample.fit_resample(trainX,trainY)
testXUnder,testYUnder = undersample.fit_resample(testX, testY)