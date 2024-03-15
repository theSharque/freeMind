import re


def get_vocabulary(all=True):
    if all:
        return [x for x in '#_0123456789абвгдеёжзийклмнопрстуфхцчшщъыьэюя,.!?:;%@()-+=[]{}']
    else:
        return [x for x in '0123456789абвгдеёжзийклмнопрстуфхцчшщъыьэюя,.!?:;%@()-+=[]{}']


def get_words(file_name='../data/russian.txt', max_len=24, min_len=1):
    vocab = get_vocabulary()[2:]
    text = (open(file_name, 'r', encoding='utf-8').read()
            .lower()
            .translate(str.maketrans('', '', '0123456789'))
            .replace('\n', ' ') + " 0 1 2 3 4 5 6 7 8 9")
    text = ''.join(filter(lambda x: (x in [' ', '\n', '\t'] or x in vocab), text))
    text = ' '.join(re.findall(r'[\w-]+|[^\w\s]', text, re.UNICODE))
    text = ' '.join([w for w in text.split() if min_len <= len(w) <= max_len])

    return text


def get_text(file_name='../data/first.txt', max_len=24, min_len=1):
    vocab = get_vocabulary()[2:]
    text = (open(file_name, 'r', encoding='utf-8').read().lower())
    text = ''.join(filter(lambda x: (x in [' ', '\n', '\t'] or x in vocab), text))
    text = text.split('\n')
    text = [' '.join(re.findall(r'[\w-]+|[^\w\s]', w, re.UNICODE)) for w in text]

    return text


def get_text_line(file_name='../data/first.txt', max_len=24, min_len=1):
    vocab = get_vocabulary()[2:]
    text = (open(file_name, 'r', encoding='utf-8').read().lower())
    text = ''.join(filter(lambda x: (x in [' ', '\n', '\t'] or x in vocab), text))
    text = ' '.join(re.findall(r'[\w-]+|[^\w\s]', text, re.UNICODE))
    text = ' '.join([w for w in text.split() if min_len <= len(w) <= max_len])

    return text
