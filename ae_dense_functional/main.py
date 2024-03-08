import os

from train_brain import train_brain
from train_encdec import train_decoder

if __name__ == '__main__':
    while True:
        train_brain()
        train_decoder()
