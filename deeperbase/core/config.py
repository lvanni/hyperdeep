#!/usr/bin/python
# -*-coding:Utf-8 -*
'''
Created on 7 déc. 2016

@author: lvanni
'''

# Back prop
#LEARNING_RATE = 1e-2
LEARNING_RATE = 1e-3
DECACY_RATE   = 0.4
BATCH_SIZE    = 32

DWIN = 20                     # Nb de mots dans une phrase
VECT_SIZE  = [100, 10, 5, 5]  # Nb de feature associées à chaque niveau (forme/lemme/code/fonction)
N_HIDDEN   = [100, 50]        # 2 couches : 100 neurones + 50 neurones

# DEFAULT PATH
DATA_PATH   = "../data/"
CORPUS_PATH = DATA_PATH + "corpus/"
NLP_PATH    = DATA_PATH + "nlp/"
DICO_PATH   = DATA_PATH + "dico/"

NLP             = "network_state"
EMBEDDING_DICO  = "embedding_dico.txt"
OCC_DICO        = "occ_dico.txt"

