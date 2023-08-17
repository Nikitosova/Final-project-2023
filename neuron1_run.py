import pickle

import numpy as np
import tensorflow

from tensorflow.keras.preprocessing.sequence import pad_sequences

with open('tokenizer_stage_1.pickle', 'rb') as file:
    tokenizer = pickle.load(file)

with open('max_sequence_length.txt', 'r') as file:
    max_sequence_length = int(file.read().strip())

model = tensorflow.keras.models.load_model('model_stage_1.keras')


# Функция для генерации текста на основе запроса
def generate_russian_text(seed_text, next_words):
    for _ in range(next_words):
        seed_seq = tokenizer.texts_to_sequences([seed_text])[0]
        padded_seq = pad_sequences([seed_seq], maxlen=max_sequence_length - 1, padding='pre')
        predicted_word_index = np.argmax(model.predict(padded_seq), axis=-1)
        predicted_word = tokenizer.index_word[predicted_word_index[0]]
        seed_text += " " + predicted_word
    return seed_text


if __name__ == '__main__':

    # Генерация текста
    input_prompt = "Перевозка груза"
    generated_text = generate_russian_text(input_prompt, next_words=10)
    print("Сгенерированный текст:", generated_text)
