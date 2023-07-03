import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense, Activation
from tensorflow.keras.optimizers import RMSprop

filepath = 'charlie.txt'

text = open(filepath,'rb').read().decode(encoding='utf-8').lower()

text = text[1000:30000]

characters = sorted(set(text))

char_to_index = dict((c,i) for i,c in enumerate(characters))
index_to_char = dict((i,c) for i,c in enumerate(characters))

SEQ_LENGTH =40
STEP_SIZE =3

sentences = []
next_character = []

for i in range(0, len(text) - SEQ_LENGTH, STEP_SIZE):
    sentences.append(text[i: i+SEQ_LENGTH])
    next_character.append(text[i+SEQ_LENGTH])

x= np.zeros((len(sentences),SEQ_LENGTH, len(characters)),dtype=bool)
y= np.zeros((len(sentences), len(characters)), dtype=bool)

for i,sentences in enumerate(sentences):
    for t,characters in enumerate(sentences):
        x[i,t,char_to_index[characters]] =1
    y[i,char_to_index[next_character[i]]] =1

model= Sequential ()
model.add(LSTM(128,input_shape = (SEQ_LENGTH, len(characters))))
model.add(Dense(len(characters)))
model.add(Activation('softmax'))

model.compile(loss= 'categorical crossentropy',optimizer = RMSprop ( lr= 0.01))
model.fit(x,y,batch_size = 256 , epochs =4)

model.save("charlie_new_song")