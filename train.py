#!/usr/bin/python
# -*-coding:Utf-8 -*
'''
Created on 7 déc. 2016

@author: lvanni
'''
import os
import time

from core.config import EMBEDDING_DICO, CORPUS_PATH, CORPUS_SIZE, CORPUS_TYPE
from core.preprocess.dico import create_dico
from core.training.main import pre_process, training


if __name__ == '__main__':
    
    t0 = time.time()
    print "Starting process..."
    
    # creation des dictionnaires (occ et embedding)
    if not os.path.exists(EMBEDDING_DICO):
        create_dico()
    
    # récupération du corpus 
    corpus = []
    for filename in os.listdir(CORPUS_PATH):
        filename_args = filename.split(".")
        try:
            if filename_args[-2] == CORPUS_SIZE and filename_args[-1] == CORPUS_TYPE:
                corpus.append(CORPUS_PATH + filename)
        except:
            continue
    
    # découpage des textes
    x_train, x_valid, x_test, y_train, y_valid, y_test = pre_process(corpus)
    
    # Training, Validation, Test
    training(x_train, x_valid, x_test, y_train, y_valid, y_test)
    
    print "finishing in :", time.time() - t0