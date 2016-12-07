# unsupervised training
# lookuptable from Collobert 
# @author Melanie Ducoffe
# @date 01/12/2015

# get_input_from_line

import numpy as np
import sys
import theano
import theano.tensor as T
from unsupervised import LookUpTrain, getParams
import pickle
import time
from blocks.utils import shared_floatx
import copy

def sub_line(line, j, dwin):
	return line[:, j:j+dwin]

def add_padding(line, paddings):
	result = np.zeros((line.shape[0], line.shape[1] + 2*len(paddings[0]))).astype(int)
	for i in range(len(line)):
		result[i] = paddings[i] + list(line[i]) + paddings[i]
	return result

def generate_incorrect_sentence(dico, correct_sentence, index):
	return correct_sentence


if __name__=="__main__":
	line=np.zeros((4, 9)).astype(int)
	line = sub_line(line, 2, 4)
	print line.shape
	line = add_padding(line, [[0],[0],[0],[0]])
	print line.shape





