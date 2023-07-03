import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense, Activation
from tensorflow.keras.optimizers import RMSprop

filepath = 'charlie.txt'

text = open(filepath,'rb').read().decode(encoding='utf-8').lower()


