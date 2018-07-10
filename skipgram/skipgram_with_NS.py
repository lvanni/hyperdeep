#!/usr/bin/python
# -*- coding: utf-8 -*-

import gensim
import numpy as np

from keras.layers import Embedding, Reshape, Activation, Input
from keras.layers.merge import Dot
from keras.models import Model
from keras.preprocessing.sequence import skipgrams

from config import EMBEDDING_DIM, NUM_EPOCHS, NEGATIVE_SAMPLES, WINDOW_SIZE
from data_helpers import tokenize

def create_vectors(corpus_file, vectors_file):

    # GENSIM METHOD    				
    sentences = gensim.models.word2vec.LineSentence(corpus_file)
    # sg defines the training algorithm. By default (sg=0), CBOW is used. Otherwise (sg=1), skip-gram is employed.
    model = gensim.models.Word2Vec(sentences, size=EMBEDDING_DIM, window=WINDOW_SIZE, min_count=0, workers=8, sg=1)

    f = open(vectors_file ,'w')
    vectors = []
    vector = '{} {}\n'.format(len(model.wv.index2word), EMBEDDING_DIM)
    vectors.append(vector)
    f.write(vector)    
    for word in model.wv.index2word:
        vector = word + " " + " ".join(str(x) for x in model.wv[word]) + "\n"
        vectors.append(vector)
        f.write(vector)
    f.flush()
    f.close()

    print("word2vec done.")

    return vectors

    
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


