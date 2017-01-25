#!/usr/bin/python
# -*-coding:Utf-8 -*
'''
Created on 7 déc. 2016

@author: lvanni
'''
import os
import sys

from core.config import EMBEDDING_DICO, NLP_PATH, DICO_PATH

from core.preprocess.dico import create_dico
from core.training.main import pre_process, training

if __name__ == '__main__':
    
    # récupération du corpus
    corpus = {}
    for filename in os.listdir(sys.argv[1]):
        if filename != "nlp" and filename != "dico":
            corpus[filename] = [sys.argv[1] + filename]
    
    # creation des dictionnaires (occ et embedding)
    if "-c" in sys.argv:
        os.system("rm -rf " + NLP_PATH)
        os.system("rm -rf " + DICO_PATH)
    
    if not os.path.exists(EMBEDDING_DICO):
        create_dico(corpus)
    else:
        print "start training from a previous state... (use command option [-c] to clean the project first)"
    
    # découpage des textes
    x_train, x_valid, x_test, y_train, y_valid, y_test = pre_process(corpus)
    
    # Training, Validation, Test
    training(x_train, x_valid, x_test, y_train, y_valid, y_test)
    
