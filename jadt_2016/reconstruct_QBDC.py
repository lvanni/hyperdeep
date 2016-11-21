##### rebuild sentence ######
import os
from contextlib import closing
import pickle
import numpy as np
from util import get_input_from_files, add_padding

def build_database(repo, dico_filename, filenames, dwin, inverse_dico):
	index = 0
	y_value = []
	x_value = []
	original_lines=[]
	with closing(open(os.path.join(repo, dico_filename), 'rb')) as f:
		dico = pickle.load(f)
	for filename in filenames:
		lines, w_lines = get_input_from_files(repo, [filename], dico)
		for line in lines:
			x_value.append(line)
			y_value.append(index)
		for w in w_lines:
			original_lines.append(w)
		if index ==0:
			index+=1
	y_value = np.asarray(y_value, dtype=int)

	# do cut
	x = [x_.astype(int) for x_ in x_value]
	y = [y_.astype(int) for y_ in y_value]

	paddings = [ [], [], [], []]
	for i in range(dwin/2):
		for i in xrange(4):
			paddings[i].append(dico[i]['PARSING'])
	paddings = np.asarray(paddings)
	#paddings = paddings.reshape((1, paddings.shape[0], paddings.shape[1]))
	x_data = [add_padding(elem, paddings) for elem in x]

	x_final=[]; y_final=[]
	recovery={}
	for original, elem, label in zip(original_lines, x_data, y):
		for i in range(elem.shape[1] -dwin):
			sentence = elem[:,i:i+dwin]
			tmp = reconstruct_sentence(sentence, inverse_dico)
			recovery[tmp]=[label, original]
	return recovery



def invert_dico(repo, output_dico):
	with closing(open(os.path.join(repo, output_dico), 'rb')) as f:
		dico = pickle.load(f)

	invert_dico = {}
	invert_dico[0] ={}; invert_dico[1] ={};invert_dico[2] ={};invert_dico[3] ={};
	for i in range(len(dico)):
		for key in dico[i]:
			invert_dico[i][dico[i][key]]=key
	return invert_dico

# load database
def author_dataset(sentence, database):
	x_value, y_value = database
	for x,y in zip(x_value, y_value):
		if np.min(sentence==x):
			return y
	return -1

def author(sentence, recovery, labels, inverse_dico):
	tmp = reconstruct_sentence(sentence, inverse_dico)
	return labels[recovery[tmp][0]]

def reconstruct_sentence(numerical_sentence, dico):
	sentence=""
	for num_channel, i in zip(numerical_sentence, range(len(numerical_sentence))):
		for num_word in num_channel:
			data = dico[i][num_word].split('\r\n')[0]
			data=data.split('\n')[0]
			sentence+=data+" "
		#sentence+="\n"
	return sentence

def convert_to_srt(sentence, dico, recovery):
	tmp = reconstruct_sentence(sentence, dico)
	if tmp not in recovery.keys():
		import pdb
		pdb.set_trace()
		print tmp
	original = recovery[tmp][1]
	strings=["","","",""]
	for i in range(len(original)):
		for j in range(4):
			strings[j]+=original[i][j].split('\r\n')[0]+" "
	return strings[0]+'\n'+strings[1]+'\n'+strings[2]+'\n'+strings[3]+'\n'


def check_sentence(repo, filenames, dwin, data, output_dico, labels, index):
	dico = invert_dico(repo, output_dico)
	"""
	with closing(open(data, 'rb')) as f:
		right, false = pickle.load(f)
	"""
	data_i = data+"_"+str(index)
	with closing(open(data_i,'rb')) as f:
		sentences = pickle.load(f)

	# find labels and whole sentence !
	recovery = build_database(repo, output_dico, filenames, dwin, dico)
	for i, sentence in zip(range(len(sentences)), sentences):
		x,y,_ = sentence
		x = x[0]; y=y[0]
		print "numero "+str(i)+" issue de "+labels[y]
		print reconstruct_sentence(x, dico)
		print "#######"
		print convert_to_srt(x, dico, recovery)
		print '\n'
	"""
	print "extraits difficiles pour le reseau :"
	for i, sentence in zip(range(len(false)), false):
		print "numero "+str(i)+" issue de "+author(sentence, recovery, labels, dico)
		print reconstruct_sentence(sentence, dico)
		print "#######"
		print convert_to_srt(sentence, dico, recovery)
		print '\n'
	"""

if __name__=="__main__":
	#data='data/sentence/relevant_sentence_H_G'
	data ='data/dico/sentence_HG_v2'
	repo='data/dico'
	output_dico="embedding_dico_H_G_v0"
	#output_dico="embedding_dico_H_S_v3"
	dwin=20
	labels=["Hollande", "DeGaulle"]
	#labels=["Hollande", "Sarkozy"]
	filenames = ['HollandeDef.cnr', 'GaulleDef.cnr']
	#filenames = ['HollandeDef.cnr', 'Sarkozy-Inter-DEF.cnr']
	check_sentence(repo, filenames, dwin, data, output_dico, labels, 12)


