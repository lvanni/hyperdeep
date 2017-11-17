#!/usr/bin/python
# -*- coding: utf-8 -*-

#np.random.seed(13)
import gensim
from keras.layers import Embedding, Reshape, Activation, Input
from keras.layers.merge import Dot
from keras.models import Model
from keras.preprocessing.sequence import skipgrams
from keras.preprocessing.text import Tokenizer

import numpy as np

"""
TRAINING SKIP-GRAM MODEL
"""
def create_vectors(corpus_file, vectors_file):
    
    corpus = open(corpus_file).readlines()
    
    corpus = [sentence for sentence in corpus if sentence.count(' ') >= 2]
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(corpus)
    V = len(tokenizer.word_index) + 1
    print("vocabulary_size: ", V)
    
    dim_embedddings = 128
    
    w_inputs = Input(shape=(1, ), dtype='int32')
    w = Embedding(V, dim_embedddings)(w_inputs)
    
    # context
    c_inputs = Input(shape=(1, ), dtype='int32')
    c  = Embedding(V, dim_embedddings)(c_inputs)
    o = Dot(axes=2)([w, c])
    o = Reshape((1,), input_shape=(1, 1))(o)
    o = Activation('sigmoid')(o)
    
    SkipGram = Model(inputs=[w_inputs, c_inputs], outputs=o)
    SkipGram.summary()
    SkipGram.compile(loss='binary_crossentropy', optimizer='adam')
    
    for _ in range(5):
        loss = 0.
        for i, doc in enumerate(tokenizer.texts_to_sequences(corpus)):
            data, labels = skipgrams(sequence=doc, vocabulary_size=V, window_size=5, negative_samples=5.)
            x = [np.array(x) for x in zip(*data)]
            y = np.array(labels, dtype=np.int32)
            if x:
                loss += SkipGram.train_on_batch(x, y)
    
        print(loss)
    
    f = open(vectors_file ,'w')
    f.write('{} {}\n'.format(V-1, dim_embedddings))
    vectors = SkipGram.get_weights()[0]
    for word, i in tokenizer.word_index.items():
        f.write('{} {}\n'.format(word, ' '.join(map(str, list(vectors[i, :])))))
    f.close()

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
    return w2v.most_similar(positive=[word])


