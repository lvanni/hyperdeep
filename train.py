#!/usr/bin/python
# -*-coding:Utf-8 -*
'''
Created on 7 déc. 2016

@author: lvanni
'''
import json
import os
import sys

from core.config import EMBEDDING_DICO, CORPUS_PATH
from core.preprocess.dico import create_dico
from core.training.main import pre_process, training


if __name__ == '__main__':
    
    # récupération du corpus
    corpus = {}
    
    # LOADING CORPUS HYPERBASE
    # sys.argv[1] => CORPUS PATH
    # sys.argv[2] => META DATA 
    if len(sys.argv) >= 3:
        if sys.argv[1][-1] != "/":
            sys.argv += "/"
        metadata = json.load(open(sys.argv[1] + "corpus.json", "r"))
    
        for key, value in metadata[sys.argv[2]].iteritems():
            if value:
                corpus[key] = []
                for text_id in value:
                    corpus[key].append(sys.argv[1] + text_id + ".tg")
    
    # LOADING CORPUS IN FOLDER : data/corpus
    else:
        for filename in os.listdir(CORPUS_PATH):
            filename_args = filename.split(".")
            filetype = filename.split(".")[-1]
            if filetype == "cnr" or filetype == "tg":
                corpus[filename_args[0]] = [CORPUS_PATH + filename]
    
    # creation des dictionnaires (occ et embedding)
    if not os.path.exists(EMBEDDING_DICO):
        create_dico(corpus)
    
    # découpage des textes
    x_train, x_valid, x_test, y_train, y_valid, y_test = pre_process(corpus)
    
    # Training, Validation, Test
    training(x_train, x_valid, x_test, y_train, y_valid, y_test)
    
