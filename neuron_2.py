import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Пример обучающей выборки с парами плохих и хороших предложений
bad_sentences = [
    "这是一个错的句子。",
    "我不知道怎么解释这个。",
    "这里有很多错误。",
    # Добавьте больше плохих предложений
]

good_sentences = [
    "这是一个正确的句子。",
    "我知道如何解释这个。",
    "这里没有错误。",
    # Добавьте больше хороших предложений
]

# Объединение плохих и хороших предложений для создания обучающей выборки
all_sentences = bad_sentences + good_sentences
labels = [0] * len(bad_sentences) + [1] * len(good_sentences)

# Токенизация текста
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_sentences)
vocab_size = len(tokenizer.word_index) + 1

# Преобразование текста в последовательности чисел (индексы слов)
sequences = tokenizer.texts_to_sequences(all_sentences)

# Заполнение последовательностей нулями до одинаковой длины
max_sequence_length = max([len(seq) for seq in sequences])
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='post')

# Создание модели
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=128, input_length=max_sequence_length))
model.add(LSTM(128))
model.add(Dense(max_sequence_length, activation='linear'))  # Линейный слой для генерации последовательности чисел

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

# Обучение модели
labels = np.array(labels)
model.fit(padded_sequences, padded_sequences, epochs=10, batch_size=64)  # Обучаем модель предсказывать коррекцию на основе входных предложений

# Сохранение модели
model.save('chinese_correction_model.h5')

print("Обучение завершено и модель сохранена.")

input_sentence = "这是一个错的句子。"

# Преобразование предложения в последовательность чисел (индексы слов)
input_sequence = tokenizer.texts_to_sequences([input_sentence])
max_sequence_length = len(input_sequence[0])
padded_sequence = pad_sequences(input_sequence, maxlen=max_sequence_length, padding='post')

# Получение предсказания от модели (скорректированное предложение)
predicted_sequence = model.predict(padded_sequence)
predicted_sentence = tokenizer.sequences_to_texts(predicted_sequence)[0]

print("Исходное предложение:", input_sentence)
print("Скорректированное предложение:", predicted_sentence)
