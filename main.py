import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense, Activation
from tensorflow.keras.optimizers import RMSprop

filepath = 'charlie.txt'

text = open(filepath,'rb').read().decode(encoding='utf-8').lower()

text = text[1000:30000]

characters = sorted(set[text])

char_to_index = dict((c,i) for i,c in enumerate(characters))
index_to_char = dict((i,c) for i,c in enumerate(characters))
