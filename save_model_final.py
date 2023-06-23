import sqlite3
import pandas as pd
import re
import pickle
import datetime
import pymorphy2
from tensorflow.keras.preprocessing.text import \
    Tokenizer  # Методы для работы с текстами и преобразования их в последовательности
import numpy as np  # Для работы с данными
import matplotlib.pyplot as plt  # Для вывода графиков
import os  # Для работы с файлами
import keras
from tensorflow.keras import utils  # Для работы с категориальными данными

from tensorflow.keras.models import Sequential  # Полносвязная модель
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Dropout  # Слои для сети

from tensorflow.keras.layers import SpatialDropout1D, BatchNormalization, Embedding, Flatten, \
    Activation  # Слои для сети
from tensorflow.keras.preprocessing.text import \
    Tokenizer  # Методы для работы с текстами и преобразования их в последовательности
from tensorflow.keras.preprocessing.sequence import pad_sequences  # Метод для работы с последовательностями

from sklearn.preprocessing import LabelEncoder  # Метод кодирования тестовых лейблов
from sklearn.model_selection import train_test_split  # Для разделения выборки на тестовую и обучающую
# from google.colab import drive # Для работы с Google Drive
from sklearn.preprocessing import LabelEncoder
from keras.layers import LSTM, Dense, Dropout, Input
global PREDICTION_TYPE

def update_table_zero():
    conn = sqlite3.connect('db.sqlite3')
    cursor = conn.cursor()
    df = pd.read_excel('dataset.xlsx')
    themes = df['Тематика письма']
    texts = df['Суть обращения']
    print(df.shape)
    cursor.execute('''Delete from main_texts''')
    for i in range(1, len(themes) - 1):
        try:
            cursor.execute('''
            INSERT INTO main_texts(text,checked_by_human,theme,text_accuracy,uploaded) Values (?,?,?,0.81,0)
    
            ''', [str(texts[i]), 0, str(themes[i])])
        except:
            print(i)
    conn.commit()
    print(123)
    return 0


# update_table_zero()

def save_model_tokenizer_by_dataset(dataset):
    conn = sqlite3.connect('db.sqlite3')
    cursor = conn.cursor()
    data = pd.read_excel(dataset)
    df = data
    df_text = df[['Суть обращения', 'Тематика письма']]
    df_text = df_text.dropna()
    duplicateRows = df_text[df_text.duplicated()]
    duplicateRows.sort_values(by='Суть обращения')
    # Анализ уникальности и дубликатов
    pd.DataFrame(df_text['Суть обращения'].value_counts().values).value_counts()

    # Приводм данные к строчному виду
    df_text['Суть обращения'] = [x.lower() for x in df_text['Суть обращения']]
    pd.DataFrame(df_text['Суть обращения'].value_counts().values).value_counts()

    # Удаляем рабочие символы и знаки припенания
    df_text['Суть обращения'] = [re.sub(r'\W+', ' ', x) for x in df_text['Суть обращения']]
    pd.DataFrame(df_text['Суть обращения'].value_counts().values).value_counts()

    # Посмотрим сколько различных слов в нашем массиве
    results = set()
    df_text['Суть обращения'].str.lower().str.split().apply(results.update)
    print('Количество слов в массиве с обращениями', len(results))

    df_text.columns = ['text', 'category']

    texts = df_text['text'].values  # Извлекаем данные всех текстов из столбца text

    classes = list(df_text['category'].values)  # Извлекаем соответствующие им значения классов (лейблов) столбца text
    print(len(classes))
    maxWordsCount = 40000  # Зададим максимальное количество слов/индексов, учитываемое при обучении текстов

    nClasses = df_text['category'].nunique() + 1  # Задаём количество классов, обращаясь к столбцу category и оставляя

    # Преобразовываем текстовые данные в числовые/векторные для обучения нейросетью
    # Для этого воспользуемся встроенной в Keras функцией Tokenizer для разбиения текста и
    # превращения в матрицу числовых значений

    tokenizer = Tokenizer(num_words=maxWordsCount, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                          lower=True, split=' ', oov_token='unknown', char_level=False)

    tokenizer.fit_on_texts(
        texts)  # "Скармливаем" наши тексты, т.е. даём в обработку методу, который соберет словарь частотности

    # Формируем матрицу индексов по принципу Bag of Words
    xAll = tokenizer.texts_to_matrix(texts)  # Каждое слово из текста нашло свой индекс в векторе длиной maxWordsCount
    # Получим список слов, входящих в наш токенизатор (при наличии установленного maxWordsCount)
    list_columns = ['статус_отступа']
    for key_word in list(tokenizer.word_index.keys())[:maxWordsCount - 1]:
        list_columns.append(key_word)
    df_1 = pd.DataFrame([xAll[0]])  # берем матричный вид первого обращения
    df_1.columns = list_columns
    # перейдем от индексов обратно к словами

    # Преобразовываем категории в векторы
    encoder = LabelEncoder()  # Вызываем метод кодирования тестовых лейблов из библиотеки sklearn
    encoder.fit(list(df_text['category'].unique()))  # Подгружаем в него категории из нашей базы
    classesEncoded = encoder.transform(classes)  # Кодируем категории
    yAll = utils.to_categorical(classesEncoded, nClasses)  # И выводим каждый лейбл в виде вектора длиной 22,
    # с 1кой в позиции соответствующего класса и нулями

    # разбиваем все данные на обучающую и тестовую выборки с помощью метода train_test_split из библиотеки sklearn
    xTrain, xVal, yTrain, yVal = train_test_split(xAll, yAll, test_size=0.2, shuffle=True)

    # Создаём полносвязную сеть
    model01 = Sequential()
    # Входной полносвязный слой
    model01.add(Dense(1000, input_dim=maxWordsCount,
                      activation="relu"))
    # Слой регуляризации Dropout
    model01.add(Dropout(0.4))
    # Второй полносвязный слой
    model01.add(Dense(1000, activation='relu'))
    # Слой регуляризации Dropout
    model01.add(Dropout(0.4))
    # Третий полносвязный слой
    model01.add(Dense(1000, activation='relu'))
    # Слой регуляризации Dropout
    model01.add(Dropout(0.4))
    # Выходной полносвязный слой
    model01.add(Dense(nClasses, activation='softmax'))

    model01.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

    # Обучаем сеть на выборке
    history = model01.fit(xTrain,
                          yTrain,
                          epochs=20,
                          batch_size=128,
                          validation_data=(xVal, yVal))

    currPred = model01.predict(xTrain[[0]])
    # Определяем номер распознанного класса для каждохо блока слов длины xLen
    currOut = np.argmax(currPred, axis=1)
    predicted_text = encoder.inverse_transform(currOut)
    model01.save('main/my_model1.h5')
    import pickle
    with open('main/tokenizer1.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    score = model01.evaluate(xVal, yVal, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    cursor.execute('''INSERT INTO  main_modelhist(result) VALUES (?)''', [score[1]])
    conn.commit()
    return 0


def relearn_model():
    conn = sqlite3.connect('db.sqlite3')
    cursor = conn.cursor()
    data = pd.DataFrame()
    data['Суть обращения'] = cursor.execute('''Select text from main_texts''').fetchall()
    data['Тематика письма'] = cursor.execute('''Select theme from main_texts''').fetchall()
    for i in range(len(data['Суть обращения'])):
        data['Суть обращения'][i] = data['Суть обращения'][i][0]

    for i in range(len(data['Тематика письма'])):
        data['Тематика письма'][i] = data['Тематика письма'][i][0]

    df = data
    df_text = df[['Суть обращения', 'Тематика письма']]
    df_text = df_text.dropna()
    duplicateRows = df_text[df_text.duplicated()]
    duplicateRows.sort_values(by='Суть обращения')
    # Анализ уникальности и дубликатов
    pd.DataFrame(df_text['Суть обращения'].value_counts().values).value_counts()

    # Приводм данные к строчному виду
    df_text['Суть обращения'] = [x.lower() for x in df_text['Суть обращения']]
    pd.DataFrame(df_text['Суть обращения'].value_counts().values).value_counts()

    # Удаляем рабочие символы и знаки припенания
    df_text['Суть обращения'] = [re.sub(r'\W+', ' ', x) for x in df_text['Суть обращения']]
    pd.DataFrame(df_text['Суть обращения'].value_counts().values).value_counts()

    # Посмотрим сколько различных слов в нашем массиве
    results = set()
    df_text['Суть обращения'].str.lower().str.split().apply(results.update)

    df_text.columns = ['text', 'category']

    # Приводм данные к сbbтрочному виду
    df_text['text'] = [x.lower() for x in df_text['text']]

    for i in range(df_text['text'].shape[0]):
        df_text['text'][i] = ' '.join(re.sub("(@[A-Za-z0-9]+)", " ", df_text['text'][i]).split())
        df_text['text'][i] = ' '.join(re.sub("\[[^()]*\]", " ", df_text['text'][i]).split())
        df_text['text'][i] = ' '.join(re.sub("[A-Za-z0-9]", " ", df_text['text'][i]).split())
    df_text['text'] = [x.lower() for x in df_text['text']]
    # Удаляем рабочие символы и знаки припенания
    df_text['text'] = [re.sub(r'\W+', ' ', x) for x in df_text['text']]
    # Посмотрим сколько различных слов в нашем массиве
    results = set()
    df_text['text'].str.lower().str.split().apply(results.update)
    print('Количество слов в массиве с обращениями', len(results))

    df_text.columns = ['text', 'category']

    texts = df_text['text'].values  # Извлекаем данные всех текстов из столбца text

    classes = list(df_text['category'].values)  # Извлекаем соответствующие им значения классов (лейблов) столбца text

    maxWordsCount = 60000  # Зададим максимальное количество слов/индексов, учитываемое при обучении текстов

    print(df_text['category'].unique())  # Выводим все уникальные значения классов

    nClasses = df_text['category'].nunique() + 1  # Задаём количество классов, обращаясь к столбцу category и оставляя

    tokenizer = Tokenizer(num_words=maxWordsCount, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                          lower=True, split=' ', oov_token='unknown', char_level=False)

    tokenizer.fit_on_texts(
        texts)  # "Скармливаем" наши тексты, т.е. даём в обработку методу, который соберет словарь частотности

    # Преобразовываем категории в векторы
    encoder = LabelEncoder()  # Вызываем метод кодирования тестовых лейблов из библиотеки sklearn
    encoder.fit(classes)  # Подгружаем в него категории из нашей базы
    classesEncoded = encoder.transform(classes)  # Кодируем категории
    print(encoder.classes_)
    # Формируем матрицу индексов по принципу Bag of Words
    xAll = tokenizer.texts_to_matrix(texts)  # Каждое слово из текста нашло свой индекс в векторе длиной maxWordsCount

    yAll = utils.to_categorical(classesEncoded, nClasses)  # И выводим каждый лейбл в виде вектора длиной 22,
    # Получим список слов, входящих в наш токенизатор (при наличии установленного maxWordsCount)
    # с 1кой в позиции соответствующего класса и нулями
    # разбиваем все данные на обучающую и тестовую выборки с помощью метода train_test_split из библиотеки sklearn
    xTrain, xVal, yTrain, yVal = train_test_split(xAll, yAll, test_size=0.2, shuffle=True)
    # Создаём полносвязную сеть
    model01 = Sequential()
    # Входной полносвязный слой
    model01.add(Dense(100, input_dim=maxWordsCount,
                      activation="relu"))
    # Слой регуляризации Dropout
    model01.add(Dropout(0.4))
    # Второй полносвязный слой
    model01.add(Dense(100, activation='relu'))
    # Слой регуляризации Dropout
    model01.add(Dropout(0.4))
    # Третий полносвязный слой
    model01.add(Dense(100, activation='relu'))
    # Слой регуляризации Dropout
    model01.add(Dropout(0.4))
    # Выходной полносвязный слой
    model01.add(Dense(nClasses, activation='softmax'))

    model01.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

    # Обучаем сеть на выборке
    history = model01.fit(xTrain,
                          yTrain,
                          epochs=20,
                          batch_size=128,
                          validation_data=(xVal, yVal))
    model01.save('main/my_model1.h5')

    with open('main/tokenizer1.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    score = model01.evaluate(xVal, yVal, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    now = datetime.datetime.now()
    cursor.execute('''INSERT INTO  main_modelhist(result,date,modeltype) VALUES (?,?,?)''', [score[1], now.strftime("%Y-%m-%d"),1])
    conn.commit()
    return 0


# relearn_model()

def predict_theme_letter(text, model1, tokenizer1, encoder1):
    with open(tokenizer1, 'rb') as f:
        tokenizer = pickle.load(f)
    text = tokenizer.texts_to_matrix([text])
    model = keras.models.load_model(model1)
    currPred = model.predict(text)
    # Определяем номер распознанного класса для каждого блока слов длины xLen
    currOut = np.argmax(currPred, axis=1)
    encoder = LabelEncoder()
    encoder.classes_ = np.load(encoder1)
    predicted_text = encoder.inverse_transform(currOut)
    return str(predicted_text[0])


#predict_theme_letter('мкмк','main/my_model.h5','main/tokenizer.pickle','main/encoder.npy')


def reload_table(model, tokenizer, encoder):
    conn = sqlite3.connect('db.sqlite3')
    cursor = conn.cursor()
    cursor.execute('''SELECT * from main_texts where checked_by_human = 0  and theme="" ''')
    result = cursor.fetchall()
    cursor.execute('''Select * from main_modelhist''')
    re2 = cursor.fetchall()  # берём данные о результатах
    try:
        re2 = re2[-1][1]  # берём последний результат
    except:
        re2 = 0  # если данных о точности модели нет, то она равна 0
    for i in range(len(result)):
        cursor.execute('''
        UPDATE main_texts
        set theme = ? and text_accuracy ==? where text = ?''',
                       [predict_theme_letter(result[i][1], model, tokenizer, encoder),re2, result[i][1]])
    conn.commit()
    return 0


def predict_theme_letter2(text, model1, tokenizer1, MAX_SEQUENCE_LENGTH =250):
    with open(tokenizer1, 'rb') as f:
        tokenizer = pickle.load(f)
    model = keras.models.load_model(model1)
    new_complaint = [text]
    seq = tokenizer.texts_to_sequences(new_complaint)
    padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
    pred = model.predict(padded)
    labels = ['Выполнение поручений правительства РФ',
              'Касательно выдачи исходных данных',
              'Касательно выдачи технических условий',
              'Касательно изменения условий перевозки',
              'Касательно организации перевозки', 'Касательно отцепки вагонов',
              'Касательно перевозки личных вещей', 'Касательно простоя вагонов',
              'Касательно согласования плана на перевозку', 'О заключении договора',
              'О предоставлении информации об осуществлении грузоперевозок',
              'О предоставлении лимитов погрузки', 'О предоставления скидки',
              'О ходе выплонения плана погрузки',
              'Об обеспечении доступа к системе ЭТРАН.',
              'Об ограничениях отгрузки пиломатериалов',
              'Об оформлении вагонов в АС ЭТРАН', 'Перевозка на особых условиях',
              'По вопросу согласования проекта Соглашения', 'Прочее',
              'о применении Тарифного руководства', 'согласование заявок ГУ']
    return labels[np.argmax(pred)]
def relearn_model2():
    conn = sqlite3.connect('db.sqlite3')
    cursor = conn.cursor()
    df = pd.DataFrame()
    df['text'] = cursor.execute('''Select text from main_texts''').fetchall()
    df['goal'] = cursor.execute('''Select theme from main_texts''').fetchall()

    df = df.dropna()
    df = df.drop_duplicates()
    df.reset_index(drop=True, inplace=True)
    for i in range(df.shape[0]):
        df['text'][i]=df['text'][i][0]
    df_text = df[['text', 'goal']]
    # Приводм данные к строчному виду
    df_text['text'] = [x.lower() for x in df_text['text']]
    pd.DataFrame(df_text['text'].value_counts().values).value_counts()

    # Удаляем рабочие символы и знаки припенания
    df_text['text'] = [re.sub(r'\W+', ' ', x) for x in df_text['text']]

    for i in range(df_text['text'].shape[0]):
        df_text['text'][i] = ' '.join(re.sub("(@[A-Za-z0-9]+)", " ", df_text['text'][i]).split())
        df_text['text'][i] = ' '.join(re.sub("\[[^()]*\]", " ", df_text['text'][i]).split())
        df_text['text'][i] = ' '.join(re.sub("[A-Za-z0-9]", " ", df_text['text'][i]).split())
    df_text['text'] = [x.lower() for x in df_text['text']]
    pd.DataFrame(df_text['text'].value_counts().values).value_counts()
    # Удаляем рабочие символы и знаки припенания
    df_text['text'] = [re.sub(r'\W+', ' ', x) for x in df_text['text']]
    # Посмотрим сколько различных слов в нашем массиве
    results = set()

    df_text['text'].str.lower().str.split().apply(results.update)

    df_text.columns = ['text', 'category']

    texts = df_text['text'].values  # Извлекаем данные всех текстов из столбца text
    print(df_text.shape)
    classes = list(df_text['category'].values)
    MAX_NB_WORDS = 50000  # максимальный словарь
    MAX_SEQUENCE_LENGTH = 250  # максимальная длина предложения
    EMBEDDING_DIM = 250
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
    tokenizer.fit_on_texts(texts)
    word_index = tokenizer.word_index
    X = tokenizer.texts_to_sequences(df_text['text'].values)  # применяем токенайзер
    X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
    Y = pd.get_dummies(df_text['category']).values
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

    model = Sequential()
    model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))  # эмбендинг на наших текстах
    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(250, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(22, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    epochs = 24
    batch_size = 256

    history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1)

    model.save('main/my_model2.h5')


    score = model.evaluate(X_test, Y_test, verbose=0)

    with open('main/tokenizer2.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    now = datetime.datetime.now()
    cursor.execute('''INSERT INTO  main_modelhist(result,date,modeltype) VALUES (?,?,?)''', [score[1], now.strftime("%Y-%m-%d"),2])
    conn.commit()
    global LABELS
    LABELS =df_text['category'].unique()
    return 0


# reload_table('main/my_model.h5','main/tokenizer.pickle','main/encoder.npy')
