#!/usr/bin/python
# -*-coding:Utf-8 -*
'''
Created on 7 déc. 2016

@author: lvanni
'''

# Back prop
LEARNING_RATE = 1e-2
DECACY_RATE   = 0.4
BATCH_SIZE    = 32

DWIN = 20                     # Nb de mots dans une phrase
VECT_SIZE  = [100, 10, 5, 5]  # Nb de feature associées à chaque niveau (forme/lemme/code/fonction)
N_HIDDEN   = [100, 50]        # 2 couches : 100 neurones + 50 neurones

CORPUS_PATH = "data/corpus/"
NLP_PATH    = "data/nlp/"
DICO_PATH   = "data/dico"

EMBEDDING_DICO  = "data/dico/embedding_dico.txt"
OCC_DICO        = "data/dico/occ_dico.txt"

