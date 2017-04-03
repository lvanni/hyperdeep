#!/usr/bin/python
# -*-coding:Utf-8 -*
'''
Created on 7 déc. 2016

@author: lvanni
'''

# Back prop
#LEARNING_RATE = 1e-2
LEARNING_RATE = 1e-2
DECACY_RATE   = 0.4 # 0.9
BATCH_SIZE    = 128

DWIN = 20                     # Nb de mots dans une phrase
VECT_SIZE  = [100, 40, 5, 5]   # Nb de feature associées à chaque niveau (forme/lemme/code/fonction)
N_HIDDEN   = [100, 50]        # 2 couches : 100 neurones + 50 neurones

# seuil pour la conservation des tokens
# len_data = [nb_forme, nb_lemme, nb_code, nb_fonction]
LEN_DATA = [5, 5, 1, 1]

# DEFAULT PATH
DATA_PATH   = "../data/"
CORPUS_PATH = DATA_PATH + "corpus/"
NLP_PATH    = DATA_PATH + "nlp/"
DICO_PATH   = DATA_PATH + "dico/"

NLP             = "network_state"
EMBEDDING_DICO  = "embedding_dico.txt"
OCC_DICO        = "occ_dico.txt"

