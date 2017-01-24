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

import numpy
import theano

from core.config import EMBEDDING_DICO, DWIN, VECT_SIZE, N_HIDDEN, NLP_PATH, \
    CORPUS_PATH, CORPUS_TYPE
from core.preprocess.dico import get_input_from_files
from core.training.lookup import LookUpTrain
from core.training.main import pre_process
import theano.tensor as T


if __name__ == '__main__':
    
    text_to_test = sys.argv[1]
    
    print "Chargement de l'embedding.................",
    x_value = []
    try:
        with closing(open(EMBEDDING_DICO, 'rb')) as f:
            dico = pickle.load(f)
    except:
        print "ERR"
        print "missing training files!"
        sys.exit(0)
    
    print  "OK"
    
    print "Chargement du réseau......................",
    # Nb mot dans le dico/corpus
    n_mot = [len(dico[i]) for i in dico.keys()]
    print  "OK"
    
    # Compte nb classes
    corpus = []
    nb_class = 0
    for filename in os.listdir(CORPUS_PATH):
        filename_args = filename.split(".")
        try:
            if filename_args[-2] == CORPUS_SIZE and filename_args[-1] == CORPUS_TYPE:
                #corpus.append(filename_args[0])
                corpus.append(CORPUS_PATH + filename)
                nb_class += 1
        except:
            continue
        
    # Natural Langage Processing
    t_nlp = LookUpTrain(DWIN, n_mot, VECT_SIZE, N_HIDDEN, n_out=nb_class)
    t_nlp.load(NLP_PATH, "network_state_0")
    
    print "Chargement du texte.......................",    
    lines, _ = get_input_from_files([text_to_test], dico)
    for line in lines:
        x_value.append(line)
    print  "OK"
    
    # Preprocessing => découpage du texte
    x_train, x_valid, x_test, tmp, tmp, tmp = pre_process([text_to_test])

    # concatener des arrays numpy
    #x_cont = numpy.concatenate([x_train, x_valid, x_test], axis=0)
    
    # Input features
    x = T.itensor3('x') 
    
    # probabilites sur un text
    probabilities = theano.function(inputs=[x], outputs=t_nlp.probabilities_text(x), allow_input_downcast=True)
    # A TESTER
    proba_list = probabilities(x_test)
    
    # sinon
    # np.mean([probabilities(x_cont[index*batch_size:(index+1)*batch_size])])
    
    # Qualité de la prédiction
    #predict_confidency = theano.function(inputs=[x], outputs=t_nlp.predict_confidency(x)[0], allow_input_downcast=True)
    
    # test predict
    # Fonction de prediction : Pour une phrase donnée, quel est le président
    predict = theano.function(inputs=[x], outputs=t_nlp.predict(x), allow_input_downcast=True)
    for test in x_test:
        print "predict :", corpus[predict([test])].replace(CORPUS_PATH, "").split(".")[0]
    
    
