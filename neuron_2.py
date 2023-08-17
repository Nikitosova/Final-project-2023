import pickle

import jieba
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split

bad_sentences = []
good_sentences = []

with open('chinese_translate.csv', encoding='utf-8') as f:
    f.readline()
    for line in f.readlines():
        d = line.split(';')
        bad_sentences.append(d[1].strip())

with open('chinese_data.csv', encoding='utf-8') as f:
    f.readline()
    for line in f.readlines():
        d = line.split(';')
        good_sentences.append(d[1].strip())

bad_sentences_processed = [" ".join(jieba.lcut(sentence)) for sentence in bad_sentences]
good_sentences_processed = [" ".join(jieba.lcut(sentence)) for sentence in good_sentences]

# Токенизация и создание словаря
tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', oov_token='<UNK>')
tokenizer.fit_on_texts(bad_sentences_processed + good_sentences_processed)

vocab_size = len(tokenizer.word_index) + 1

# Создание и обучение модели
max_sequence_length = max(max(len(sentence.split()) for sentence in bad_sentences_processed),
                          max(len(sentence.split()) for sentence in good_sentences_processed))

print(max_sequence_length)
# Создание и обучение модели
input_layer = Input(shape=(max_sequence_length,))
embedding_layer = Embedding(vocab_size, 128)(input_layer)
lstm_layer = LSTM(256, return_sequences=True)(embedding_layer)
output_layer = Dense(vocab_size, activation='softmax')(lstm_layer)

model = Model(inputs=input_layer, outputs=output_layer)
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

bad_sequences = tokenizer.texts_to_sequences(bad_sentences_processed)
bad_sequences_padded = tf.keras.preprocessing.sequence.pad_sequences(bad_sequences, maxlen=max_sequence_length, padding='post')

good_sequences = tokenizer.texts_to_sequences(good_sentences_processed)
good_sequences_padded = tf.keras.preprocessing.sequence.pad_sequences(good_sequences, maxlen=max_sequence_length, padding='post')

X_train, X_val, y_train, y_val = train_test_split(bad_sequences_padded, good_sequences_padded, test_size=0.1,
                                                  random_state=43)

model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=50, epochs=30)

# Сохранение модели
model.save('model_stage_2.keras')

with open('tokenizer_stage_2.pickle', 'wb') as file:
    pickle.dump(tokenizer, file)

with open('max_sequence_length_stage_2.txt', 'w') as file:
    file.write(str(max_sequence_length))

print("Обучение завершено и модель сохранена.")
