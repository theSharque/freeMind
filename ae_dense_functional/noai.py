import random

import dictionary

word2int = {}
int2word = {}
seq = {}
context = ''


def load_seq(n: int):
    global seq, context
    text = dictionary.get_text_line('../data/harry.txt').split()
    context = ' '.join(text[:7])

    for i in range(len(text) - n):
        key = ' '.join(text[i:i + n])
        if key in seq:
            seq[key].append(text[i + n])
        else:
            seq[key] = [text[i + n], ]


def load_all_seq():
    global context
    load_seq(1)
    load_seq(2)
    load_seq(3)
    load_seq(4)
    load_seq(5)
    load_seq(6)


def generator():
    global seq, context
    while True:
        words = context.split()
        short = words.pop()
        while len(seq[short]) > 2 and len(words) > 1:
            pre_short = ' '.join([words.pop(), short])
            if pre_short in seq:
                short = pre_short
            else:
                break

        word = seq[short][random.randint(0, len(seq[short]) - 1)]
        print(word, end=' ')
        context = ' '.join(context.split()[:6]) + ' ' + word


if __name__ == '__main__':
    load_all_seq()
    print(context, end=' ')
    generator()
