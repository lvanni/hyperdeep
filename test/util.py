# util function : parsing and dictionnary
# @author melanie ducoffe
from contextlib import closing
import os
import pickle
import numpy as np
import copy

# parse a document using dot
# read line by line

def generate_incorrect_sentence(dico, correct_sentence, index):
	incorrect_sentence = copy.copy(correct_sentence)
	nb_words = [len(dico[i].keys()) for i in range(len(dico))]
	new_indices = []
	i=0
	while (i <len(correct_sentence)):
		value = np.random.randint(nb_words[i])
		if value != correct_sentence[i,index]:
			new_indices.append(index)
			incorrect_sentence[i][index]=value
			i+=1
	return incorrect_sentence
		
def sub_line(line, j, dwin):
	line_sub = np.zeros((line.shape[0], dwin))
	for i in range(line.shape[0]):
		line_sub[i]=line[i, j:j+dwin]
	return line_sub

#add_padding
def add_padding(line, paddings):
	return np.concatenate([paddings, line, paddings], axis=1)

def get_input_from_file(repo, filename, dico, padding=[]):
	sentences=[]
	word_sentences=[]
	with closing(open(os.path.join(repo, filename), 'rb')) as f:
		line = []
		rare = [dico[0]['RARE'], dico[1]['RARE'],
			dico[2]['RARE'], dico[3]['RARE']]
		for words in f:
			tags = words.split('\t')[2:]
			tags = [tag.lower() for tag in tags]
			if len(tags)==0:
				continue
			line.append(tags)
			if tags[0]=='.':
				lemme = np.zeros((4, len(line))).astype(int)
				for i in range(len(line)):
					if len(line[i])<4:
						continue
					lemme[0][i] = dico[0].get(line[i][0], rare[0])
					lemme[1][i] = dico[1].get(line[i][1], rare[1])
					lemme[2][i] = dico[2].get(line[i][2], rare[2])
					lemme[3][i] = dico[3].get(line[i][3], rare[3])
				sentences.append(lemme)
				word_sentences.append(line)
				line = []

	return sentences, word_sentences

def get_input_from_files(repo, filenames, dico):
	sentences=[]
	for filename in filenames :
		sentences += get_input_from_file(repo, filename, dico)

	return sentences

def get_input_from_line(repo, filename):
	sentences_tags=[]
	with closing(open(os.path.join(repo, filename), 'rb')) as f:
		for line in f:
			if len(line)<=2:
				continue
			sequence = line.split('\t')[2:]
			sequence = [elem.lower() for elem in sequence]
			sentences_tags.append(sequence)
	return sentences_tags

def build_dico_from_file(repo, filename, dico):
	# get lines
	sentences_tags = get_input_from_line(repo, filename) # open in universal format !!!!
	for line in sentences_tags:
		for i in range(len(dico.keys())):
			if len(line) < len(dico.keys()):
				continue
			if len(line[i])==0:
				continue
			if line[i] not in dico[i].keys():
				dico[i][line[i]]= 1
			else:
				dico[i][line[i]]+=1
	return dico

def build_dictionnary(repo, filenames, output_filename):
	dico ={}
	dico[0]={}; dico[1]={}; dico[2]={}; dico[3]={}
	index = [0, 0, 0, 0]
	for filename in filenames:
		print filename
		dico = build_dico_from_file(repo, filename, dico)

	with closing(open(os.path.join(repo, output_filename), 'wb')) as f:
		pickle.dump(dico, f, protocol=pickle.HIGHEST_PROTOCOL)
	return dico


def build_dico_from_occ(repo, filename_dico, output_dico):
	with closing(open(os.path.join(repo, filename_dico), 'rb')) as f:
		dico_occ = pickle.load(f)
	# afficher le nombre de cles avec plus de 1000 occurences
	occ = [0,0,0,0]
	#len_data=[1, 5, 500, 500]
	len_data = [50,50,1,1]
	"""
	for i in range(4):
		for elem in dico_occ[i]:
			if dico_occ[i][elem] >= len_data[i]:
				occ[i]+=1.
		occ[i]/=len(dico_occ[i])
		occ[i]*=100
	print occ
	import pdb
	pdb.set_trace()
	print 'kikou'
	"""
	index = [0, 0, 0, 0]
	dico = {}; dico[0]={}; dico[1]={}; dico[2]={}; dico[3]={}
	# RARE PARSING
	for i in range(4):
		dico[i]['RARE'] = index[i]	
		dico[i]['PARSING']=index[i]+1
		index[i] = 2

	for i in range(4):
		for elem in dico_occ[i]:
			#elem = elem.lower()
			if dico_occ[i][elem] >=len_data[i] and elem.lower() not in dico[i]:
				elem = elem.lower()
				dico[i][elem] = index[i]
				index[i]+=1

	with closing(open(os.path.join(repo, output_dico), 'wb')) as f:
		pickle.dump(dico, f, protocol = pickle.HIGHEST_PROTOCOL)
			
	return dico

def buid_dataset_lookup(repo, filenames, dico, dwin):
	print("Loading Data ... \n")
	paddings = [ [], [], [], []]
	for i in range(dwin/2):
		for i in xrange(4):
			paddings[i].append(dico[i]['PARSING'])
	paddings = np.asarray(paddings)
	input_sentences = get_input_from_files(repo, filenames, dico)
	input_sentences = [ input_sentences[i] for i in np.random.permutation(len(input_sentences))]
	# random shuffle and use 10 % of the data for validation
	index_train = int(len(input_sentences)*0.9)

	data_train = []
	data_train_c = []
	index = 1
	for epoch in range(2):

		for line in input_sentences:
			# add dwin/2 blanck space before and after the sentence
			line = add_padding(line, paddings)
			sentence_size = len(line[0])
			# select randomly 3 j
			for j in np.random.permutation(sentence_size -dwin)[:10]: ##parcours tous les lemmes de la phrase moins le padding : -dwin
				# on ne considere qu'une fenetre de taille dwin
				correct_sentence = sub_line(line, j, dwin)
				# select randomly word and labels
				incorrect_sentence = generate_incorrect_sentence(dico, correct_sentence, dwin/2)
				data_train.append(correct_sentence)
				data_train_c.append(incorrect_sentence)

			if index % 1000 == 0:
				data_save = os.path.join(repo, 'lookup_data_savings_HG_v0'+str(index))
				with closing(open(data_save, 'wb')) as f:
					pickle.dump([data_train, data_train_c], f, protocol = pickle.HIGHEST_PROTOCOL)
			index +=1
		data_save = os.path.join(repo, 'lookup_data_savings_HG_v0'+str(index))
		with closing(open(data_save, 'wb')) as f:
			pickle.dump([data_train, data_train_c], f, protocol = pickle.HIGHEST_PROTOCOL)
			index +=1

def decode_dico():
	print 'NOT IMPLEMENTED'

if __name__=="__main__":
	"""
	filenames = ['GaulleDef.cnr', 'Chirac1def.cnr', 'Chirac2Def.cnr', 'GISCARDDef.cnr',
			'HollandeDef.cnr', 'PompidouDef.cnr', 'Mitterrand1DEF.cnr', 'Mitterrand2Def.cnr',
			'Sarkozy-Inter-DEF.cnr']
	"""
	#filenames = ['HollandeDef.cnr', 'Sarkozy-Inter-DEF.cnr']
	filenames = ['HollandeDef.cnr', 'GaulleDef.cnr']
	repo = './data/dico'
	filename_dico ='dico_occ_Hollande_Gaulle'
	#dico = build_dictionnary(repo, filenames, filename_dico)
	output_dico ="embedding_dico_H_G_v0"
	dico = build_dico_from_occ(repo, filename_dico, output_dico)
	print len(dico[0])
	print len(dico[1])
	print len(dico[2])
	print len(dico[3])
	#buid_dataset_lookup(repo, filenames, dico, 20)
	#lines = get_input_from_files(repo, filenames, dico)
	# cut to build the database for the lookuptable
	#import pdb
	#pdb.set_trace()
	print 'end'
