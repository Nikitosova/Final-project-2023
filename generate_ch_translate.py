import time

from googletrans import Translator

# Создание объекта Translator
translator = Translator()

data = []

ch_f = open('chinese_translate.csv', 'a', encoding='utf-8')

with open('Russian_data.csv', encoding='utf-8') as f:
    f.readline()
    i = 0
    for line in f.readlines():
        d = line.split(';')
        if i % 50 == 0:
            time.sleep(0.5)
        if len(d) == 2:
            translate_ok = False
            while not translate_ok:
                try:
                    text = d[1]
                    translated = translator.translate(text, src='ru', dest='zh-cn')
                    line = d[0] + ';' + translated.text + '\n'
                    print(text)
                    print(translated.text)
                    print()
                    i += 1
                    ch_f.write(line)
                    translate_ok = True
                except:
                    time.sleep(2)

ch_f.close()
