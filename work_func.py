import time

import pandas as pd

import json

from scipy.spatial.distance import cosine

from help_funcs import lemmatize, create_tf_idf

df = pd.read_csv('processed_data_OK.csv', index_col='index', encoding='utf-32')
df_ch = pd.read_csv('Chinese_data.csv', encoding='utf-8', sep=';')
df_ru = pd.read_csv('Russian_data.csv', encoding='utf-8', sep=';')
# подгружаем словарь с пронумерованными словами
with open('words_indices_OK.json', 'r') as f:
    words_indices = json.load(f)

# подгружаем словарь idf
with open('idf_data_OK.json', 'r') as f:
    idf = json.load(f)

df['tf_idf'] = df['tf_idf'].apply(json.loads)


def get_best_sentences(user_text: str, limit=10):
    ts = time.time()
    lemmatize_text = lemmatize(user_text).split()
    good_words = []
    for word in lemmatize_text:
        if word in idf:
            good_words.append(word)

    if len(good_words) == 0:
        return []

    my_tf_idf = create_tf_idf(idf, [good_words], words_indices)[0]

    good_i = {}
    for i, val in enumerate(my_tf_idf):
        if val > 0:
            good_i[i] = val

    result = []
    for i, row in df.iterrows():
        check = False
        tf_idf = row['tf_idf']
        for g_i in good_i:
            if tf_idf[g_i] > 0:
                check = True
                break

        if check:
            score = cosine(tf_idf, my_tf_idf)
            if 0 < score < 1:
                result.append([score, row['index_sentence']])

    result.sort(key=lambda x: x[0])

    indexes = []
    data = []

    for score, index in result:
        indexes.append(index)

    for i in indexes:
        s_ru = df_ru.iloc[i].text
        s_ch = df_ch.iloc[i].text

        d = (s_ru, s_ch)
        if d not in data:
            data.append(d)

        if len(data) >= limit:
            break

    return data


if __name__ == '__main__':
    # тестовый пример
    from pprint import pprint

    text = 'груз, контейнер'
    data = get_best_sentences(text, 5)
    pprint(data)
