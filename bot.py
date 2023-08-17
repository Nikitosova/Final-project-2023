from telebot import TeleBot

from work_func import get_best_sentences

token = 'insert your token here'
bot = TeleBot(token)

hello_message = 'Приветствую тебя в моём боте.' \
                'Я помогу тебе разобраться в логистике.'


@bot.message_handler(commands=['start'])
def main_handler(message):
    user_id = message.from_user.id
    bot.send_message(user_id, hello_message)


@bot.message_handler()
def main_handler(message):
    user_id = message.from_user.id
    user_text = message.text
    print('Запрос', user_text, user_id)
    data = get_best_sentences(user_text)
    text = 'Лучшие предложения:\n'
    for rus, ch in data:
        text += f'{rus}\n{ch}\n\n'
    bot.send_message(user_id, text)


print('Запускаю бота')
bot.infinity_polling()
