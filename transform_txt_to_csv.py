data = []


def clear_russian_text(text: str):
    new_text = text.replace(',', '').replace('.', '')
    return new_text


def clear_chinese_text(text: str):
    new_text = text
    return new_text


with open('RussianChinese_dataset.txt', encoding='utf-8') as f:
    russian = None
    chinese = None
    for line in f.readlines():
        d = line.split(':', 1)
        if d[0] == 'Russian':
            russian = d[1].strip()
        elif d[0] == 'Chinese':
            chinese = d[1].strip()

        if russian is not None and chinese is not None:
            russian = clear_russian_text(russian)
            chinese = clear_chinese_text(chinese)
            data.append((russian, chinese))
            russian = None
            chinese = None
        elif line == '\n':
            russian = None
            chinese = None
rus_sens = []
title = 'sentence_index;text\n'
with open('Russian_data.csv', 'w', encoding='utf-8') as f_ru:
    with open('Chinese_data.csv', 'w', encoding='utf-8') as f_ch:
        i = 0
        f_ru.write(title)
        f_ch.write(title)

        for russian, chinese in data:
            f_ru.write(f'{i};{russian}\n')
            f_ch.write(f'{i};{chinese}\n')
            i += 1
            rus_sens.append(russian)

with open('rus_sens.txt', 'w', encoding='utf-8') as f:
    for sens in rus_sens:
        f.write(sens + '\n')

