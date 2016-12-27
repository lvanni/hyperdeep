#!/usr/bin/python
# -*-coding:Utf-8 -*
'''
Created on 7 déc. 2016

@author: lvanni
'''
import os

from core.config import EMBEDDING_DICO, CORPUS_PATH
from core.preprocess.dico import create_dico
from core.training.main import pre_process, training

if __name__ == '__main__':
    
    print "Starting process..."
    
    # creation des dictionnaires (occ et embedding)
    if not os.path.exists(EMBEDDING_DICO):
        create_dico()
    
    # récupération du corpus 
    corpus = []
    for filename in os.listdir(CORPUS_PATH):
        if ".txt" in filename:
            corpus.append(CORPUS_PATH + filename)
    
    # découpage des textes
    x_train, x_valid, x_test, y_train, y_valid, y_test = pre_process(corpus)
    
    # Training, Validation, Test
    training(x_train, x_valid, x_test, y_train, y_valid, y_test)