#!/usr/bin/python
# -*- coding: utf-8 -*-

import gensim
import numpy as np

from keras.layers import Embedding, Reshape, Activation, Input
from keras.layers.merge import Dot
from keras.models import Model
from keras.preprocessing.sequence import skipgrams
from keras.preprocessing.text import Tokenizer

from config import EMBEDDING_DIM, MAX_NB_WORDS, NUM_EPOCHS, NEGATIVE_SAMPLES, WINDOW_SIZE

def create_vectors(corpus_file, vectors_file=False):
    
    corpus = open(corpus_file).readlines()
    
    corpus = [sentence for sentence in corpus if sentence.count(' ') >= 2]
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(corpus)
    V = len(tokenizer.word_index) + 1
    print("vocabulary_size: ", V)
    
    w_inputs = Input(shape=(1, ), dtype='int32')
    w = Embedding(V, EMBEDDING_DIM)(w_inputs)
    
    # context
    c_inputs = Input(shape=(1, ), dtype='int32')
    c  = Embedding(V, EMBEDDING_DIM)(c_inputs)
    o = Dot(axes=2)([w, c])
    o = Reshape((1,), input_shape=(1, 1))(o)
    o = Activation('sigmoid')(o)
    
    SkipGram = Model(inputs=[w_inputs, c_inputs], outputs=o)
    SkipGram.summary()
    SkipGram.compile(loss='binary_crossentropy', optimizer='adam')
    
    for epoch in range(NUM_EPOCHS):
        loss = 0.
        for i, doc in enumerate(tokenizer.texts_to_sequences(corpus)):
            data, labels = skipgrams(sequence=doc, vocabulary_size=V, window_size=WINDOW_SIZE, negative_samples=NEGATIVE_SAMPLES)
            x = [np.array(x) for x in zip(*data)]
            y = np.array(labels, dtype=np.int32)
            if x:
                loss += SkipGram.train_on_batch(x, y)
    
        print("Loss : ", loss/epoch)

    vectors = SkipGram.get_weights()[0]
    w2v = []
    if not vectors_file:
        vectors_file = corpus_file + ".vec"
    f = open(vectors_file ,'w')
    f.write('{} {}\n'.format(V-1, EMBEDDING_DIM))
    for word, i in tokenizer.word_index.items():
        vector = '{} {}\n'.format(word, ' '.join(map(str, list(vectors[i, :]))))
        f.write(vector)
        w2v += [vector]
    f.close()

    return w2v
    
"""
GET W2VEC MODEL
"""
def get_w2v(vectors_file):
    return gensim.models.KeyedVectors.load_word2vec_format(vectors_file, binary=False)

"""
GET WORD VECTOR
"""
def get_vector(word, w2v):
    pass  
 
"""
FIND MOST SIMILAR WORD
"""
def get_most_similar(word, vectors_file):
    w2v = get_w2v(vectors_file)
    most_similar = w2v.most_similar(positive=[word])
    print(most_similar)
    return most_similar


