#######
#nouveau fichier de training#
#merde !!!!#
#######

from unsupervised import LookUpTable, Window, LookUpTrain, getParams
import theano
import theano.tensor as T
import numpy 
from training_unsupervised import (sub_line, add_padding,
				generate_incorrect_sentence)
from util import build_dictionnary, get_input_from_files
import pickle
import time
from blocks.utils import shared_floatx
import copy
from contextlib import closing
import os
	
def build_embedding_data(repo, filenames, database_name, filename_load, filename_save):
	# build the network
	dico, _ = build_dictionnary(repo, filenames)
	dwin = 9
	paddings = [ [], [], [], []]
	"""
	for i in range(dwin/2):
		for i in xrange(4):
			paddings[i].append(dico[i]['PADDING'])
	"""
	paddings = numpy.asarray(paddings)
	"""
	# parametres et creation de LookUpTrain :
	n_mot = [len(dico[i]) for i in dico.keys()]
	vect_size = [1000, 1000, 5, 5]
	n_hidden = 25
	x = T.imatrix('x')
	t_nlp = LookUpTrain(dwin, n_mot, vect_size, n_hidden)
	t_nlp.initialize()
	lookup = theano.function(inputs=[x], outputs=t_nlp.embedding(x), allow_input_downcast=True)
	"""
	# load lines
	input_sentences = get_input_from_files(repo, filenames, dico, paddings)
	embedding = []
	#t_nlp.load(repo, filename_load)
	total = 0
	for line in input_sentences:
		np_line = numpy.zeros((4, len(line[0])))
		np_line[0] = line[0]
		np_line[1] = line[1]
		np_line[2] = line[2]
		np_line[3] = line[3]
		np_line = np_line.astype(int)
		#latent_variables = lookup(np_line)
		embedding.append(np_line)

	path = os.path.join(repo, filename_save+"_good")

	with closing(open(path, 'wb')) as f:
		pickle.dump(embedding, f, protocol=pickle.HIGHEST_PROTOCOL)



def unsupervised_training(learning_rate, decay_rate, epochs, repo, output_dico, database_name):
	dwin = 9
	with closing(open(os.path.join(repo, output_dico), 'rb')) as f:
		dico = pickle.load(f)
	n_mot = [len(dico[i]) for i in dico.keys()]
	vect_size = [20, 10, 5, 5]
	n_hidden = 100
	x = T.itensor3('x')
	xc = T.itensor3('x')
	y = T.ivector('y')
	#xc = T.itensor3('xc')
	t_nlp = LookUpTrain(dwin, n_mot, vect_size, n_hidden)
	t_nlp.initialize()
	cost = T.mean(t_nlp.cost(x, y))
	error = T.mean(t_nlp.errors(x,y))

	params = getParams(t_nlp, x)
	for p, i in zip(params, range(len(params))):
		p.name+='_'+str(i)

	#calcul du gradient avec RMSProp
	updates = []
	caches = {}
	grad_params = T.grad(cost, params)
	for param, grad_param in zip(params, grad_params):

		if not caches.has_key(param.name):
			caches[param.name] = shared_floatx(param.get_value() * 0.,
												"cache_"+param.name)
		# update rule
		update_cache = decay_rate*caches[param.name]\
					+ (1 - decay_rate)*grad_param**2
		update_param = param  - learning_rate*grad_param/T.sqrt(update_cache + 1e-8)
		updates.append((caches[param.name], update_cache))
		updates.append((param, update_param))

	train_model = theano.function(inputs=[x,y], outputs=cost, updates=updates,
					allow_input_downcast=True)

	valid_model = theano.function(inputs=[x, y], outputs=cost, allow_input_downcast=True)
	test_model = theano.function(inputs=[x, y], outputs=error, allow_input_downcast=True)
	data_path = os.path.join(repo, database_name)
	with closing(open(data_path, 'rb')) as f:
		data, data_c = pickle.load(f)
	data = numpy.asarray(data).astype(int)
	labels = numpy.asarray(data_c).astype(int)

	# test : reduce data
	data = data
	data_c = data_c
	# reading by minibatch
	batch_size = 15
	n_sample = data.shape[0]/batch_size
	# 80% of the data will go into the training set
	n_train = (int) (n_sample*0.8)

	y_value = numpy.zeros((2*batch_size),dtype=int)
	y_value[batch_size:]= 1+ y_value[batch_size:]
	index_filename = 0
	
	saving = "params_savings_bis_v4_"
	
	#t_nlp.load(repo, (saving+str(95)))
	#index_filename = 96
	#saving = "params_savings_bis"

	for epoch in range(4):
		train_cost = []
		valid_cost = []
		index_valid = n_train
		for minibatch_index in range(n_train):

			correct_sentences = data[minibatch_index*batch_size:(minibatch_index+1)*batch_size, :, :]
			incorrect_sentences = data_c[minibatch_index*batch_size:(minibatch_index+1)*batch_size, :, :]
			sentences = numpy.concatenate([incorrect_sentences, correct_sentences], axis=0)
			train_value = train_model(sentences, y_value)
			if minibatch_index%10==0:
				train_cost=[]
				for minibatch_train in range(n_train):
					correct_sentences = data[minibatch_train*batch_size:(minibatch_train+1)*batch_size, :, :]
					incorrect_sentences = data_c[minibatch_train*batch_size:(minibatch_train+1)*batch_size, :, :]
					sentences = numpy.concatenate([incorrect_sentences, correct_sentences], axis=0)
					train_value = valid_model(sentences, y_value)
					train_cost.append(train_value)
				print "Train : "+str(numpy.mean(train_cost)*100)
				valid_cost=[]
				for minibatch_valid in range(n_train, n_sample):
					correct_sentences = data[minibatch_valid*batch_size:(minibatch_valid+1)*batch_size, :, :]
					incorrect_sentences = data_c[minibatch_valid*batch_size:(minibatch_valid+1)*batch_size, :, :]
					sentences = numpy.concatenate([incorrect_sentences, correct_sentences], axis=0)
					valid_value = test_model(sentences, y_value)
					#import pdb
					#pdb.set_trace()
					valid_cost.append(valid_value)
				print "Valid : "+str(numpy.mean(valid_value)*100)+" in : "+(saving+str(index_filename))
				t_nlp.save(repo, (saving+str(index_filename)))
				index_filename+=1

if __name__ == '__main__':
	learning_rate=1e-2
	decay_rate = 0.9
	epochs = 100
	repo='data/dico'
	output_dico ="embedding_dico_H_S_v3"
	#filename_load = 'params_savings_HS'
	database_name="lookup_data_savings_HS_v327064"
	unsupervised_training(learning_rate, decay_rate, epochs, repo, output_dico, database_name)
