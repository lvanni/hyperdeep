#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
Created on 16 nov. 2017
@author: laurent.vanni@unice.fr
'''
import sys

from classifier.cnn.main import train
from skipgram.skipgram_with_NS import create_vectors, get_most_similar


def print_help():
    print("usage: python hyperdeep.py <command> <args>\n")
    print("The commands supported by deeperbase are:\n")
    print("\tskipgram\ttrain a skipgram model")
    print("\tnn\t\tquery for nearest neighbors\n")
    print("\ttrain\ttrain a CNN model for sentence classification")
    
def print_invalidArgs_mess():
    print("Invalid argument detected!\n")

def get_args():
    args = {}
    for i in range(2, len(sys.argv[1:])+1):
        if sys.argv[i][0] == "-":
            args[sys.argv[i]] = sys.argv[i+1]
        else:
            args[i] = sys.argv[i]
    return args

if __name__ == '__main__':

    # GET COMMAND
    try:
        command = sys.argv[1]
        if command not in ["skipgram", "nn", "train"]:
            raise
    except:
        print_help()
        exit()

    

    # EXECT COMMAND
    if command == "skipgram":
        try:
            args = get_args()
            corpus_file = args["-input"]
            vectors_file = args["-output"]
            create_vectors(corpus_file, vectors_file)
        except:
            print_invalidArgs_mess()
            print("The following arguments are mandatory:\n")
            print("\t-input\ttraining file path")
            print("\t-output\toutput file path\n")
            print_help()
            exit()
            
    if command == "train":
        try:
            args = get_args()
            corpus_file = args["-input"]
            model_file = args["-output"]
            train(corpus_file, model_file, args.get("-w2vec", False))
        except:
            raise
            print_invalidArgs_mess()
            print("The following arguments are mandatory:\n")
            print("\t-input\ttraining file path")
            print("\t-output\toutput file path\n")
            print("The following arguments for training are optional:\n")
            print("\t-w2vec\tword vector representations file path\n")
            print_help()
            exit()
            
    if command == "nn": # nearest neighbors
        try:
            args = get_args()
            print(args)
            model = args[2]
            word = args[3]
            most_similar_list = get_most_similar(word, model)
            print(most_similar_list)
        except:
            print_invalidArgs_mess()
            print("usage: python hyperdeep.py nn <model> <word>\n")
            print("\tmodel\ttmodel filename")
            print("\tword\tinput word\n")
            print_help()
            exit()
            
