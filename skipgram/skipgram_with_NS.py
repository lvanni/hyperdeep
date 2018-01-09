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
    
    texts = open(corpus_file, "r").readlines()
    sentences = []
    for line in texts:
        sentences += line.split()
    my_dictionary, data = tokenize(texts, vectors_file.replace(".vec", ""), True)   				
				
    V = len(my_dictionary["word_index"]) + 1
    print("vocabulary_size: ", V)

    # GENSIM METHOD    				
    sentences = gensim.models.word2vec.LineSentence(corpus_file)
    # sg defines the training algorithm. By default (sg=0), CBOW is used. Otherwise (sg=1), skip-gram is employed.
    model = gensim.models.Word2Vec(sentences, size=EMBEDDING_DIM, window=WINDOW_SIZE, min_count=5, workers=8, sg=1)

    f = open(vectors_file ,'w')
    f.write('{} {}\n'.format(len(model.wv.index2word), EMBEDDING_DIM))    
    for word in model.wv.index2word:
        f.write(word + " " + " ".join(str(x) for x in model.wv[word]) + "\n")
    f.close()
				
    return 0
    
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


