import pickle

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas


data = []
with open('rus_sens.txt', encoding='utf-8') as f:
    for s in f.readlines():
        data.append(s.strip())


# Токенизация текста
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data)
vocab_size = len(tokenizer.word_index) + 1

# Преобразование текста в последовательности чисел (индексы слов)
sequences = tokenizer.texts_to_sequences(data)
max_sequence_length = max([len(seq) for seq in sequences])

# Подготовка входных и выходных данных
input_sequences = []
output_sequences = []
for sequence in sequences:
    for i in range(1, len(sequence)):
        input_sequences.append(sequence[:i])
        output_sequences.append(sequence[i])

input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_length - 1, padding='pre')
output_sequences = np.array(output_sequences)

# Создание модели
model = Sequential()
model.add(Embedding(vocab_size, 50, input_length=max_sequence_length - 1))
model.add(LSTM(100))
model.add(Dense(vocab_size, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(input_sequences, output_sequences, epochs=50, verbose=2)

model.save('model_stage_1.keras')

with open('tokenizer_stage_1.pickle', 'wb') as file:
    pickle.dump(tokenizer, file)

with open('max_sequence_length.txt', 'w') as file:
    file.write(str(max_sequence_length))


