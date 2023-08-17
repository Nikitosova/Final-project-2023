import pickle

import jieba
import numpy as np
import tensorflow

with open('tokenizer_stage_2.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

with open('max_sequence_length_stage_2.txt', 'r') as file:
    max_sequence_length = int(file.read().strip())

model = tensorflow.keras.models.load_model('model_stage_2.keras')


def correct_ch_sentence(input_sentence):
    input_sentences_processed = [" ".join(jieba.lcut(input_sentence))]

    input_sequences = tokenizer.texts_to_sequences(input_sentences_processed)
    input_sequences_padded = tensorflow.keras.preprocessing.sequence.pad_sequences(input_sequences,
                                                                                   maxlen=max_sequence_length,
                                                                                   padding='post')

    corrected_sequences = model.predict(input_sequences_padded)

    corrected_sentences = []
    for sequence_probs in corrected_sequences:
        predicted_indices = [np.argmax(probabilities) for probabilities in sequence_probs]
        corrected_sentence = " ".join(
            [tokenizer.index_word.get(idx, '') if idx != 0 else '' for idx in predicted_indices]
        )
        corrected_sentences.append(corrected_sentence)

    return corrected_sentences[0].strip()


if __name__ == '__main__':
    inp_sentence = "这是一个错的句子。"
    predicted = correct_ch_sentence(inp_sentence)
    print("Исходное предложение:", inp_sentence)
    print("Скорректированное предложение:", predicted)
