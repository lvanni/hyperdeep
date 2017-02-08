#!/usr/bin/python
# -*-coding:Utf-8 -*
'''
Created on 7 déc. 2016

@author: lvanni
'''
from collections import OrderedDict
from contextlib import closing
import os
import pickle

from deeperbase.core.config import DICO_PATH, OCC_DICO, EMBEDDING_DICO, NLP_PATH, CORPUS_PATH, NLP
from core.preprocess.dico import create_dico
from core.training.main import pre_process, training

if __name__ == '__main__':
    
    # récupération du corpus
    corpus = {}
    for filename in os.listdir(CORPUS_PATH):
        try:
            file_type = filename.split(".")[1].lower()
            if file_type == "txt" or file_type == "tg" or file_type == "cnr":
                corpus[filename] = [CORPUS_PATH + filename]
        except:
            continue
    corpus = OrderedDict(sorted(corpus.items(), key=lambda t: t[0]))
    
    if not os.path.exists(DICO_PATH) or not os.path.exists(DICO_PATH + OCC_DICO) or not os.path.exists(DICO_PATH + EMBEDDING_DICO):
        # CREATE DICO PATH
        if not os.path.exists(DICO_PATH):
            os.makedirs(DICO_PATH)
        
        # create dico
        occ_dico, embedding_dico = create_dico(corpus)
        
        # CREATE OCC_DICO PATH
        with closing(open(DICO_PATH + OCC_DICO, 'wb')) as f:
            pickle.dump(occ_dico, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # CREATE EMBEDDING_DICO PATH
        with closing(open(DICO_PATH + EMBEDDING_DICO, 'wb')) as f:
            pickle.dump(embedding_dico, f, protocol = pickle.HIGHEST_PROTOCOL)
    else:
        # on charge le dico
        with closing(open(DICO_PATH + EMBEDDING_DICO, 'rb')) as f:
            embedding_dico = pickle.load(f)
        print "start training from a previous state... (use command option [-c] to clean the project first)"
    
    # decoupage des textes
    x_train, x_valid, x_test, y_train, y_valid, y_test = pre_process(corpus, embedding_dico)
    
    # Training, Validation, Test
    t_nlp = training(x_train, x_valid, x_test, y_train, y_valid, y_test, embedding_dico)
    
    # saving network state
    if not os.path.exists(NLP_PATH):
        os.makedirs(NLP_PATH)
    t_nlp.save(NLP_PATH + NLP)
    
