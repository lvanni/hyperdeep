#!/usr/bin/python
# -*-coding:Utf-8 -*
'''
Created on 7 déc. 2016

@author: lvanni
'''
from contextlib import closing
import os
import pickle
import sys

from core.config import NLP, DICO_PATH
import theano

from core.config import EMBEDDING_DICO, DWIN, VECT_SIZE, N_HIDDEN, NLP_PATH
from core.training.lookup import LookUpTrain
from core.training.main import pre_process
import theano.tensor as T


if __name__ == '__main__':
    
    # récuperation du texte
    corpus = {}
    corpus["text"] = [sys.argv[1]]
    
    print("Chargement de l'embedding................."),
    try:
        with closing(open(DICO_PATH + EMBEDDING_DICO, 'rb')) as f:
            dico = pickle.load(f)
    except:
        print("ERR")
        print("missing training files!")
        sys.exit(0)
    
    print  ("OK")
    
    print ("Chargement du réseau......................"),
    # Nb mot dans le dico/corpus
    n_mot = [len(dico[i]) for i in dico.keys()]
    print  ("OK")
    
    # Natural Langage Processing
    t_nlp = LookUpTrain(DWIN, n_mot, VECT_SIZE, N_HIDDEN)
    
    # get the last networkstate
    netstate = "network_state"
    for filename in os.listdir(NLP_PATH):
        filename = filename.replace(NLP_PATH, "")
        if filename > netstate:
            netstate = filename
    t_nlp.load(NLP_PATH + NLP)
    
    # Preprocessing => découpage du texte
    x_train, x_valid, x_test, tmp, tmp, tmp = pre_process(corpus, dico)

    # concatener des arrays numpy
    #x_cont = numpy.concatenate([x_train, x_valid, x_test], axis=0)
    
    # Input features
    x = T.itensor3('x') 
    
    # probabilites sur un text
    probabilities = theano.function(inputs=[x], outputs=t_nlp.probabilities_text(x), allow_input_downcast=True)
    
    proba_list = probabilities(x_test)
    print (proba_list)    
    
