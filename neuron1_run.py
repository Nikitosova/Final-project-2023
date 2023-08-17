# Функция для генерации текста на основе запроса
def generate_text(seed_text, next_words):
    for _ in range(next_words):
        seed_seq = tokenizer.texts_to_sequences([seed_text])[0]
        padded_seq = pad_sequences([seed_seq], maxlen=max_sequence_length - 1, padding='pre')
        predicted_word_index = np.argmax(model.predict(padded_seq), axis=-1)
        predicted_word = tokenizer.index_word[predicted_word_index[0]]
        seed_text += " " + predicted_word
    return seed_text


# Генерация текста
input_prompt = "Перевозка груза"
generated_text = generate_text(input_prompt, next_words=10)
print("Сгенерированный текст:", generated_text)
