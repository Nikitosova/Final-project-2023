# -*- coding: utf-8 -*-
.

"""

# !pip install pandas
# !pip install -U scikit-learn
# !pip install nltk
# !pip install pymorphy2

import pandas as pd
import os

from pymorphy2 import MorphAnalyzer
import nltk
from nltk.tokenize import word_tokenize, wordpunct_tokenize
from nltk import download
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn
import numpy as np

from math import log
from collections import Counter

import json

print(sklearn.__version__)


download('punkt')
download('stopwords')

from itertools import count

i = count()


def lemmatize(a_text):
    try:
        print(next(i))
        a_tokens = wordpunct_tokenize(a_text)
        a_lemmatized = [morph.parse(item)[0].normal_form for item in a_tokens]
        a_lemmatized = ' '.join([token for token in a_lemmatized if token.isalpha()])
        return a_lemmatized
    except Exception as e:
        return ''


def get_top_tf_idf_words(tfidf_vector, feature_names, top_n):
    sorted_nzs = np.argsort(tfidf_vector.data)[:-(top_n + 1):-1]
    return feature_names[tfidf_vector.indices[sorted_nzs]]


def add_best_words(row):
    i = row.name
    article_vector = articles_tfidf[i, :]
    words = get_top_tf_idf_words(article_vector, feature_names, 10)
    return words


def enumerate_words(lists_of_keywords):
    i = 0
    words_indices = {}
    for keywords in lists_of_keywords:
        for word in keywords:
            if word not in words_indices:
                words_indices[word] = i
                i += 1

    return words_indices


def create_idf(lists_of_keywords):
    count_dictionary = Counter()
    for keywords in lists_of_keywords:
        words = set(keywords)
        for word in words:
            count_dictionary[word] += 1

    idf = {}
    for word, count in count_dictionary.items():
        idf[word] = log(len(lists_of_keywords) / count)
        
    return idf


def create_tf_idf(idf, lists_of_keywords, words_indices):
    tf_idf = [[0] * len(idf) for i in range(len(lists_of_keywords))]
    for n, sentence in enumerate(lists_of_keywords):
        words_counter = Counter(sentence)
        sentence_length = len(sentence)
        for word in words_counter:
            word_index = words_indices[word]
            word_tf = words_counter[word] / sentence_length
            word_tf_idf = word_tf * idf[word]

            tf_idf[n][word_index] = word_tf_idf

    return tf_idf


# Читаем необработанный корпус, где 3 колонки
df = pd.read_csv('Russian_data.csv', sep=';', encoding='utf-8')
print(df)

# Создаём стоп слова и анализатор
stops = stopwords.words('russian')
morph = MorphAnalyzer()

# Создаём колонку с обработанным текстом
df['lemmatized_text'] = df.text.apply(lemmatize)

# Создаём векторизатор
tfidf = TfidfVectorizer(
    analyzer="word",  # анализировать по словам или по символам (char)
    stop_words=stops,  # передаём список стоп-слов для русского из NLTK
    ngram_range=(1, 1)
)

articles_tfidf = tfidf.fit_transform(df['lemmatized_text'])

feature_names = np.array(tfidf.get_feature_names_out())
feature_words = []
for i, article in enumerate(df.lemmatized_text):
    article_vector = articles_tfidf[i, :]
    words = get_top_tf_idf_words(article_vector, feature_names, 5)

# Создаём колонку с ключевыми словами
df['keywords'] = df.apply(add_best_words, axis=1)

words_indices = enumerate_words(df.keywords)


idf = create_idf(df['keywords'])
tf_idf = create_tf_idf(idf, df['keywords'], words_indices)

# Создаём колонку с tf_idf векторами
df['tf_idf'] = tf_idf

# Сохранение словаря в файл
with open('words_indices_OK.json', 'w') as f:
    json.dump(words_indices, f)

# Сохранение словаря в файл
with open('idf_data_OK.json', 'w') as f:
    json.dump(idf, f)

# Чтение соваря из файла
with open('words_indices_OK.json', 'r') as f:
    words_indices_loaded = json.load(f)

df.to_csv('processed_data_OK.csv', index_label='index', encoding='utf-32')

df
