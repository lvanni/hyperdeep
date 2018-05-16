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
    model = gensim.models.Word2Vec(sentences, size=EMBEDDING_DIM, window=WINDOW_SIZE, min_count=5, workers=8, sg=1)

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

def create_tg_vectors(corpus_file, vectors_file):

    print("CREATE TG vectors")

    corpus = open(corpus_file, "r").readlines()
    forme = open(corpus_file + ".FORME", "w")
    code = open(corpus_file + ".CODE", "w")
    lemme = open(corpus_file + ".LEMME", "w")

    for line in corpus:
        for word in line.split():
            if "__" in word : continue
            try:
                args = word.split("**")
                forme.write(args[0] + " ")
                code.write(args[1] + " ")
                lemme.write(args[2] + " ")
            except:
                forme.write("PAD ")
                code.write("PAD ")
                lemme.write("PAD ")
        forme.write("\n")
        code.write("\n")
        lemme.write("\n")
    forme.close()
    code.close()
    lemme.close()

    vectors_tg = []
    for ext in [".FORME", ".CODE", ".LEMME"]:
        v = {}
        sentences = gensim.models.word2vec.LineSentence(corpus_file + ext)
        model = gensim.models.Word2Vec(sentences, size=int(EMBEDDING_DIM/3), window=WINDOW_SIZE, min_count=0, workers=8, sg=1)
        for word in model.wv.index2word:
            v[word] = " ".join(str(x) for x in model.wv[word])
        vectors_tg.append(v)

    vectors = {}
    corpus = open(corpus_file, "r").readlines()
    for line in corpus:
        for word in line.split():
            if "__" in word or vectors.get(word, False): continue
            if word == "PAD":    
                vectors[word] = "PAD " + "0 " * EMBEDDING_DIM + "\n"
                continue
            
            # make vectors representation
            args = word.split("**")            
            v = word + " "
            i = 0
            for arg in args:
                try:
                    v += vectors_tg[i][arg] + " "
                except:
                    v += "0 " * int(EMBEDDING_DIM/3) + " "
                i += 1
            v += "\n"
            vectors[word] = v
            
            # TODO ajouter les representaiton partielle : 
            #      FROME**PAD**PAD
            #      PAD**CODE**PAD
            #      PAD**PAD**LEMME
            #vectors[args[0]] = args[0] + " " + vectors_tg[0][args[0]] + " " + "0 " * int((EMBEDDING_DIM/3)*2) + "\n"            
            #vectors[args[1]] = args[1] + " " + "0 " * int(EMBEDDING_DIM/3) + " " + vectors_tg[1][args[1]] + " " + "0 " * int(EMBEDDING_DIM/3) + "\n"            
            #vectors[args[2]] = args[2] + " " + "0 " * int((EMBEDDING_DIM/3)*2) + vectors_tg[2][args[2]] + " " + "\n"
            
            

    voc_size = len(vectors.keys())
    vectors = list(vectors.values())

    # VECTOR HEADER
    vector = ['{} {}\n'.format(voc_size, int(EMBEDDING_DIM/3)*3)]
    vectors = vector + vectors

    f = open(vectors_file ,'w')
    for vector in vectors:
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


