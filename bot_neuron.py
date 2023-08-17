from telebot import TeleBot

from neuron1_run import generate_russian_text
from googletrans import Translator

from neuron2_run import correct_ch_sentence

token = '6698275913:AAH9oHA5RQG_7HE8V81x7lS_Vl_iddXepic'
bot = TeleBot(token)

hello_message = 'Приветствую тебя в моём боте.' \
                'Я помогу тебе разобраться в логистике.'

translator = Translator()


@bot.message_handler(commands=['start'])
def main_handler(message):
    user_id = message.from_user.id
    bot.send_message(user_id, hello_message)


@bot.message_handler()
def main_handler(message):
    user_id = message.from_user.id
    user_text = message.text

    message = bot.send_message(user_id, 'Генерирую текст, подождите...')
    new_rus_text = generate_russian_text(user_text, next_words=10)

    text = 'Сгенерированный текст:\n' + new_rus_text + '\n'
    bot.edit_message_text(text + 'Перевожу текст...', user_id, message.message_id)

    attempt = 0
    success = False
    translated = None
    while attempt < 5:
        try:
            translated = translator.translate(new_rus_text, src='ru', dest='zh-cn').text
            success = True
            break
        except:
            attempt += 1

    if success:
        corrected = correct_ch_sentence(translated)
        text += f'Машинный перевод:\n{translated}\nСкорректированный перевод:\n{corrected}'
    else:
        text += f'Не удалось выполнить машинный перевод'
    bot.edit_message_text(text, user_id, message.message_id)


print('Запускаю бота')
bot.infinity_polling()
